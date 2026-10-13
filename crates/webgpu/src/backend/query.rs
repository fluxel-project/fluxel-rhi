//! Native WebGPU occlusion query sets and direct command lowering.
//!
//! WebGPU ties occlusion queries to the pass descriptor. The encoder records
//! that identity at pass begin and rejects attempts to begin a query from a
//! different set while the pass is live.

use std::ops::Range;

use fluxel_rhi_core::api::{self, Error, Result};
use js_sys::Object;
use wasm_bindgen::JsValue;

use super::{
    command::WebGpuEncoder,
    instance::{WebGpu, WebGpuDevice},
    js,
    resource::WebGpuBuffer,
};

/// The browser query-set object together with the count validated by Core.
#[derive(Clone)]
pub struct WebGpuQuerySet {
    pub(super) raw: JsValue,
    count: u32,
}

fn error(value: JsValue) -> Error {
    Error::Backend(js::message(&value))
}

fn descriptor() -> JsValue {
    Object::new().into()
}

fn field(object: &JsValue, name: &str, value: JsValue) -> Result<()> {
    js::set(object, name, &value).map_err(error)
}

fn valid_slots(set: &WebGpuQuerySet, slots: Range<u32>) -> Result<()> {
    if slots.start >= slots.end || slots.end > set.count {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

impl api::QueryBackend for WebGpu {
    fn query_supported(_: &WebGpuDevice, kind: api::QueryKind) -> bool {
        matches!(kind, api::QueryKind::Occlusion)
    }

    fn create_query_set(device: &WebGpuDevice, desc: &api::QueryDesc) -> Result<WebGpuQuerySet> {
        if !matches!(desc.kind, api::QueryKind::Occlusion) || desc.count == 0 {
            return Err(Error::Unsupported("WebGPU query type/count"));
        }
        let native = descriptor();
        field(&native, "type", JsValue::from_str("occlusion"))?;
        field(&native, "count", JsValue::from_f64(desc.count as f64))?;
        let raw = js::call1(&device.0.device, "createQuerySet", &native).map_err(error)?;
        Ok(WebGpuQuerySet {
            raw,
            count: desc.count,
        })
    }

    fn write_timestamp(_: &mut WebGpuEncoder, _: &WebGpuQuerySet, _: u32) -> Result<()> {
        Err(Error::Unsupported("WebGPU timestamps"))
    }

    fn begin_occlusion(encoder: &mut WebGpuEncoder, set: &WebGpuQuerySet, slot: u32) -> Result<()> {
        if slot >= set.count {
            return Err(Error::InvalidRange);
        }
        if encoder.occlusion_active {
            return Err(Error::InvalidState);
        }
        let pass = encoder.pass.as_ref().ok_or(Error::InvalidState)?;
        if !encoder
            .occlusion_query_set
            .as_ref()
            .is_some_and(|active| Object::is(active, &set.raw))
        {
            return Err(Error::InvalidState);
        }
        js::call1(pass, "beginOcclusionQuery", &JsValue::from_f64(slot as f64)).map_err(error)?;
        encoder.occlusion_active = true;
        encoder.keep_alive.push(set.raw.clone());
        Ok(())
    }

    fn reset_queries(_: &mut WebGpuEncoder, set: &WebGpuQuerySet, slots: Range<u32>) -> Result<()> {
        // WebGPU query slots are reset implicitly when a command buffer uses
        // them.  The range still needs backend-side validation.
        valid_slots(set, slots)
    }

    fn end_occlusion(encoder: &mut WebGpuEncoder) -> Result<()> {
        if !encoder.occlusion_active {
            return Err(Error::InvalidState);
        }
        let pass = encoder.pass.as_ref().ok_or(Error::InvalidState)?;
        js::call0(pass, "endOcclusionQuery").map_err(error)?;
        encoder.occlusion_active = false;
        Ok(())
    }

    fn begin_statistics(_: &mut WebGpuEncoder, _: &WebGpuQuerySet, _: u32) -> Result<()> {
        Err(Error::Unsupported("WebGPU pipeline statistics"))
    }

    fn end_statistics(_: &mut WebGpuEncoder) -> Result<()> {
        Err(Error::Unsupported("WebGPU pipeline statistics"))
    }

    fn resolve_queries(
        encoder: &mut WebGpuEncoder,
        set: &WebGpuQuerySet,
        slots: Range<u32>,
        destination: &WebGpuBuffer,
        offset: u64,
    ) -> Result<()> {
        valid_slots(set, slots.clone())?;
        if encoder.pass.is_some()
            || !destination.usage.contains(api::BufferUsage::QUERY_RESOLVE)
            || offset % 256 != 0
        {
            return Err(Error::InvalidRange);
        }
        let bytes = u64::from(slots.end - slots.start)
            .checked_mul(8)
            .ok_or(Error::InvalidRange)?;
        if offset
            .checked_add(bytes)
            .is_none_or(|end| end > destination.size)
        {
            return Err(Error::InvalidRange);
        }
        js::call(
            &encoder.raw,
            "resolveQuerySet",
            &[
                set.raw.clone(),
                JsValue::from_f64(slots.start as f64),
                JsValue::from_f64((slots.end - slots.start) as f64),
                destination.raw.clone(),
                JsValue::from_f64(offset as f64),
            ],
        )
        .map_err(error)?;
        encoder
            .keep_alive
            .extend([set.raw.clone(), destination.raw.clone()]);
        Ok(())
    }
}
