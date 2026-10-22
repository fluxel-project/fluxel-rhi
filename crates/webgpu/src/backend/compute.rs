//! Direct browser compute pass and pipeline lowering.

use std::{future::Future, pin::Pin};

use fluxel_rhi_core::api::{self, Error, Result};
use js_sys::{Array, Object, Promise};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

use super::{
    command::WebGpuEncoder,
    instance::{WebGpu, WebGpuDevice},
    js,
    resource::WebGpuBuffer,
};

/// WebGPU exposes compute on the same native queue as raster work.
pub struct WebGpuCompute;

fn error(value: JsValue) -> Error {
    Error::Backend(js::message(&value))
}

fn field(object: &JsValue, name: &str, value: JsValue) -> Result<()> {
    js::set(object, name, &value).map_err(error)
}

fn object() -> JsValue {
    Object::new().into()
}

impl api::ComputeOps<WebGpu> for WebGpuCompute {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = Pin<Box<dyn Future<Output = Result<JsValue>> + 'a>>;

    fn create_pipeline<'a>(
        device: &'a WebGpuDevice,
        desc: &'a api::ComputePipelineDesc<WebGpu>,
    ) -> Self::PipelineFuture<'a> {
        Box::pin(async move {
            let pipeline = object();
            // SAFETY: Core's `ComputeOps` entry validates the layout and shader
            // DeviceIds against `device` before this backend call.
            let layout = unsafe { desc.layout.native_ref() }.clone();
            let immediate_info = super::immediate::layout_info(&layout);
            field(&pipeline, "layout", layout.clone())?;
            if let Some(label) = &desc.label {
                field(&pipeline, "label", JsValue::from_str(label))?;
            }
            let stage = object();
            // SAFETY: Core validated this shader's DeviceId against `device`;
            // the clone is retained by the browser pipeline descriptor.
            let module = unsafe { desc.compute.module.native_ref() }.clone();
            if desc.required_subgroup_size.is_some()
                && !js::has_feature(&device.0.device, "subgroup-size-control")
            {
                return Err(Error::Unsupported("WebGPU subgroup-size-control feature"));
            }
            let module = super::immediate::variant_module(
                device,
                &module,
                immediate_info[2],
                &desc.compute.entry,
                desc.required_subgroup_size
                    .map(|size| (desc.compute.entry.as_str(), size)),
            )?;
            field(&stage, "module", module)?;
            field(&stage, "entryPoint", JsValue::from_str(&desc.compute.entry))?;
            if !desc.compute.constants.is_empty() {
                field(
                    &stage,
                    "constants",
                    super::pipeline::constants(&desc.compute.constants)?,
                )?;
            }
            field(&pipeline, "compute", stage)?;
            let promise = js::call1(&device.0.device, "createComputePipelineAsync", &pipeline)
                .map_err(error)?;
            let result = JsFuture::from(Promise::from(promise))
                .await
                .map_err(|error| Error::Shader(js::message(&error)))?;
            super::immediate::associate_pipeline(&result, &layout);
            Ok(result)
        })
    }

    fn begin(encoder: &mut WebGpuEncoder) -> Result<()> {
        if encoder.pass.is_some() {
            return Err(Error::InvalidState);
        }
        encoder.pass = Some(js::call1(&encoder.raw, "beginComputePass", &object()).map_err(error)?);
        Ok(())
    }

    fn set_pipeline(encoder: &mut WebGpuEncoder, pipeline: &JsValue) -> Result<()> {
        let pass = encoder.pass.as_ref().ok_or(Error::InvalidState)?;
        js::call1(pass, "setPipeline", pipeline).map_err(error)?;
        super::immediate::bind_pipeline(pass, pipeline, true)?;
        encoder.keep_alive.push(pipeline.clone());
        Ok(())
    }

    fn set_bind_group(
        encoder: &mut WebGpuEncoder,
        slot: u32,
        group: &JsValue,
        offsets: &[u32],
    ) -> Result<()> {
        let pass = encoder.pass.as_ref().ok_or(Error::InvalidState)?;
        let dynamic = Array::new();
        for offset in offsets {
            dynamic.push(&JsValue::from_f64(*offset as f64));
        }
        js::call(
            pass,
            "setBindGroup",
            &[
                JsValue::from_f64(slot as f64),
                group.clone(),
                dynamic.into(),
            ],
        )
        .map_err(error)?;
        encoder.keep_alive.push(group.clone());
        Ok(())
    }

    fn set_push_constants(encoder: &mut WebGpuEncoder, offset: u32, bytes: &[u8]) -> Result<()> {
        let pass = encoder.pass.as_ref().ok_or(Error::InvalidState)?;
        super::immediate::write(pass, api::StageMask::COMPUTE, offset, bytes)
    }

    fn dispatch(encoder: &mut WebGpuEncoder, groups: [u32; 3]) -> Result<()> {
        let pass = encoder.pass.as_ref().ok_or(Error::InvalidState)?;
        js::call(
            pass,
            "dispatchWorkgroups",
            &groups.map(|n| JsValue::from_f64(n as f64)),
        )
        .map_err(error)?;
        Ok(())
    }

    fn dispatch_indirect(
        encoder: &mut WebGpuEncoder,
        args: &WebGpuBuffer,
        offset: u64,
    ) -> Result<()> {
        if !args.usage.contains(api::BufferUsage::INDIRECT)
            || offset % 4 != 0
            || offset.checked_add(12).is_none_or(|end| end > args.size)
        {
            return Err(Error::InvalidRange);
        }
        let pass = encoder.pass.as_ref().ok_or(Error::InvalidState)?;
        js::call(
            pass,
            "dispatchWorkgroupsIndirect",
            &[args.raw.clone(), JsValue::from_f64(offset as f64)],
        )
        .map_err(error)?;
        encoder.keep_alive.push(args.raw.clone());
        Ok(())
    }

    fn end(encoder: &mut WebGpuEncoder) -> Result<()> {
        let pass = encoder.pass.take().ok_or(Error::InvalidState)?;
        js::call0(&pass, "end").map_err(error)?;
        Ok(())
    }

    fn abort(encoder: &mut WebGpuEncoder) {
        encoder.pass.take();
    }
}
