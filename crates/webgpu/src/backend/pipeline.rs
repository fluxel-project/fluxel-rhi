//! WGSL shader and browser raster pipeline construction.

use std::{
    future::{Future, Ready, ready},
    pin::Pin,
};

use fluxel_rhi_core::api::{self, Error, Result};
use js_sys::{Array, Object, Promise};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

use super::{
    instance::{WebGpu, WebGpuDevice},
    js,
};

fn error(value: JsValue) -> Error {
    Error::Backend(js::message(&value))
}

fn object() -> JsValue {
    Object::new().into()
}

fn field(object: &JsValue, name: &str, value: JsValue) -> Result<()> {
    js::set(object, name, &value).map_err(error)
}

impl api::ShaderBackend for WebGpu {
    type DirectFuture<'a> = Ready<Result<JsValue>>;
    type NagaFuture<'a> = Ready<Result<JsValue>>;

    fn shader_route(_: &WebGpuDevice, desc: &api::ShaderDesc) -> api::ShaderRoute {
        match desc.format {
            api::ShaderFormat::Wgsl => api::ShaderRoute::Direct(api::ShaderTarget {
                format: api::ShaderFormat::Wgsl,
                environment: "browser-webgpu".into(),
                abi_key: "wgsl-group-binding".into(),
            }),
            api::ShaderFormat::SpirV => api::ShaderRoute::Naga(api::ShaderTarget {
                format: api::ShaderFormat::Wgsl,
                environment: "browser-webgpu".into(),
                abi_key: "spirv-to-wgsl-group-binding".into(),
            }),
            _ => api::ShaderRoute::Unsupported,
        }
    }

    fn compile_direct<'a>(
        device: &'a WebGpuDevice,
        desc: &'a api::ShaderDesc,
        _: &'a api::ShaderTarget,
    ) -> Ready<Result<JsValue>> {
        let result = (|| {
            let source = std::str::from_utf8(&desc.bytes)
                .map_err(|error| Error::Shader(format!("WGSL UTF-8: {error}")))?;
            let native = object();
            field(&native, "code", JsValue::from_str(source))?;
            if let Some(label) = &desc.label {
                field(&native, "label", JsValue::from_str(label))?;
            }
            js::call1(&device.0.device, "createShaderModule", &native).map_err(error)
        })();
        ready(result)
    }

    fn compile_naga<'a>(
        device: &'a WebGpuDevice,
        desc: &'a api::ShaderDesc,
        _: &'a api::ShaderTarget,
    ) -> Ready<Result<JsValue>> {
        let result = (|| {
            let module = naga::front::spv::parse_u8_slice(
                &desc.bytes,
                &naga::front::spv::Options {
                    // Fluxel's scene matrices already apply the framebuffer
                    // coordinate convention for these checked-in shaders.
                    adjust_coordinate_space: false,
                    ..Default::default()
                },
            )
            .map_err(|error| Error::Shader(format!("SPIR-V parse: {error}")))?;
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .map_err(|error| Error::Shader(format!("SPIR-V validation: {error}")))?;
            let source = naga::back::wgsl::write_string(
                &module,
                &info,
                naga::back::wgsl::WriterFlags::empty(),
            )
            .map_err(|error| Error::Shader(format!("WGSL emission: {error}")))?;
            let native = object();
            field(&native, "code", JsValue::from_str(&source))?;
            if let Some(label) = &desc.label {
                field(&native, "label", JsValue::from_str(label))?;
            }
            js::call1(&device.0.device, "createShaderModule", &native).map_err(error)
        })();
        ready(result)
    }
}

impl api::PipelineBackend for WebGpu {
    type RasterPipelineFuture<'a> = Pin<Box<dyn Future<Output = Result<JsValue>> + 'a>>;

    fn create_raster_pipeline<'a>(
        device: &'a WebGpuDevice,
        desc: &'a api::RasterPipelineDesc<Self>,
    ) -> Self::RasterPipelineFuture<'a> {
        Box::pin(async move {
            let native = raster_descriptor(desc)?;
            let promise =
                js::call1(&device.0.device, "createRenderPipelineAsync", &native).map_err(error)?;
            JsFuture::from(Promise::from(promise))
                .await
                .map_err(|error| Error::Shader(js::message(&error)))
        })
    }
}

fn raster_descriptor(desc: &api::RasterPipelineDesc<WebGpu>) -> Result<JsValue> {
    if desc.multiview.is_some() || desc.primitive.unclipped_depth || desc.primitive.conservative {
        return Err(Error::Unsupported("WebGPU raster pipeline option"));
    }
    if !matches!(desc.primitive.polygon_mode, api::PolygonMode::Fill) {
        return Err(Error::Unsupported("WebGPU polygon mode"));
    }
    if !desc.vertex.constants.is_empty()
        || desc
            .fragment
            .as_ref()
            .is_some_and(|f| !f.constants.is_empty())
    {
        return Err(Error::Unsupported("WebGPU pipeline constants"));
    }
    let native = object();
    field(
        &native,
        "layout",
        unsafe { desc.layout.native_ref() }.clone(),
    )?;
    if let Some(label) = &desc.label {
        field(&native, "label", JsValue::from_str(label))?;
    }

    let vertex = object();
    field(
        &vertex,
        "module",
        unsafe { desc.vertex.module.native_ref() }.clone(),
    )?;
    field(&vertex, "entryPoint", JsValue::from_str(&desc.vertex.entry))?;
    let buffers = Array::new();
    for buffer in &desc.vertex_buffers {
        let layout = object();
        field(
            &layout,
            "arrayStride",
            JsValue::from_f64(buffer.stride as f64),
        )?;
        field(
            &layout,
            "stepMode",
            JsValue::from_str(match buffer.step_mode {
                api::VertexStepMode::Vertex => "vertex",
                api::VertexStepMode::Instance => "instance",
            }),
        )?;
        let attributes = Array::new();
        for attribute in &buffer.attributes {
            let value = object();
            field(
                &value,
                "shaderLocation",
                JsValue::from_f64(attribute.location as f64),
            )?;
            field(&value, "offset", JsValue::from_f64(attribute.offset as f64))?;
            field(
                &value,
                "format",
                JsValue::from_str(vertex_format(attribute.format)?),
            )?;
            attributes.push(&value);
        }
        field(&layout, "attributes", attributes.into())?;
        buffers.push(&layout);
    }
    field(&vertex, "buffers", buffers.into())?;
    field(&native, "vertex", vertex)?;

    if let Some(fragment) = &desc.fragment {
        let part = object();
        field(
            &part,
            "module",
            unsafe { fragment.module.native_ref() }.clone(),
        )?;
        field(&part, "entryPoint", JsValue::from_str(&fragment.entry))?;
        let targets = Array::new();
        for target in &desc.color_targets {
            let Some(target) = target else {
                targets.push(&JsValue::NULL);
                continue;
            };
            let value = object();
            field(
                &value,
                "format",
                JsValue::from_str(super::resource::texture_format(target.format)?),
            )?;
            field(
                &value,
                "writeMask",
                JsValue::from_f64(target.write_mask as f64),
            )?;
            if let Some(blend) = target.blend {
                field(&value, "blend", blend_state(blend)?)?;
            }
            targets.push(&value);
        }
        field(&part, "targets", targets.into())?;
        field(&native, "fragment", part)?;
    }

    let primitive = object();
    field(
        &primitive,
        "topology",
        JsValue::from_str(match desc.primitive.topology {
            api::PrimitiveTopology::PointList => "point-list",
            api::PrimitiveTopology::LineList => "line-list",
            api::PrimitiveTopology::LineStrip => "line-strip",
            api::PrimitiveTopology::TriangleList => "triangle-list",
            api::PrimitiveTopology::TriangleStrip => "triangle-strip",
        }),
    )?;
    field(
        &primitive,
        "frontFace",
        JsValue::from_str(match desc.primitive.front_face {
            api::FrontFace::Ccw => "ccw",
            api::FrontFace::Cw => "cw",
        }),
    )?;
    field(
        &primitive,
        "cullMode",
        JsValue::from_str(match desc.primitive.cull_mode {
            api::CullMode::None => "none",
            api::CullMode::Front => "front",
            api::CullMode::Back => "back",
        }),
    )?;
    field(&native, "primitive", primitive)?;
    if let Some(depth) = &desc.depth_stencil {
        if !matches!(
            depth.format,
            api::TextureFormat::Depth16Unorm
                | api::TextureFormat::Depth24Plus
                | api::TextureFormat::Depth32Float
        ) {
            return Err(Error::Unsupported("WebGPU stencil pipeline lowering"));
        }
        let state = object();
        field(
            &state,
            "format",
            JsValue::from_str(super::resource::texture_format(depth.format)?),
        )?;
        field(
            &state,
            "depthWriteEnabled",
            JsValue::from_bool(depth.depth_write_enabled),
        )?;
        field(
            &state,
            "depthCompare",
            JsValue::from_str(compare(depth.depth_compare)),
        )?;
        field(
            &state,
            "depthBias",
            JsValue::from_f64(depth.depth_bias_constant as f64),
        )?;
        field(
            &state,
            "depthBiasSlopeScale",
            JsValue::from_f64(depth.depth_bias_slope_scale as f64),
        )?;
        field(
            &state,
            "depthBiasClamp",
            JsValue::from_f64(depth.depth_bias_clamp as f64),
        )?;
        field(&native, "depthStencil", state)?;
    }
    let multisample = object();
    field(
        &multisample,
        "count",
        JsValue::from_f64(desc.multisample.count as f64),
    )?;
    field(
        &multisample,
        "mask",
        JsValue::from_f64((desc.multisample.mask as u32) as f64),
    )?;
    field(
        &multisample,
        "alphaToCoverageEnabled",
        JsValue::from_bool(desc.multisample.alpha_to_coverage_enabled),
    )?;
    field(&native, "multisample", multisample)?;
    Ok(native)
}

fn blend_state(state: api::BlendState) -> Result<JsValue> {
    let value = object();
    field(&value, "color", blend_component(state.color)?)?;
    field(&value, "alpha", blend_component(state.alpha)?)?;
    Ok(value)
}

fn blend_component(component: api::BlendComponent) -> Result<JsValue> {
    let value = object();
    field(
        &value,
        "srcFactor",
        JsValue::from_str(blend_factor(component.src_factor)?),
    )?;
    field(
        &value,
        "dstFactor",
        JsValue::from_str(blend_factor(component.dst_factor)?),
    )?;
    field(
        &value,
        "operation",
        JsValue::from_str(blend_operation(component.operation)),
    )?;
    Ok(value)
}

fn blend_factor(factor: api::BlendFactor) -> Result<&'static str> {
    Ok(match factor {
        api::BlendFactor::Zero => "zero",
        api::BlendFactor::One => "one",
        api::BlendFactor::SrcColor => "src",
        api::BlendFactor::OneMinusSrcColor => "one-minus-src",
        api::BlendFactor::DstColor => "dst",
        api::BlendFactor::OneMinusDstColor => "one-minus-dst",
        api::BlendFactor::SrcAlpha => "src-alpha",
        api::BlendFactor::OneMinusSrcAlpha => "one-minus-src-alpha",
        api::BlendFactor::DstAlpha => "dst-alpha",
        api::BlendFactor::OneMinusDstAlpha => "one-minus-dst-alpha",
        api::BlendFactor::Constant => "constant",
        api::BlendFactor::OneMinusConstant => "one-minus-constant",
        api::BlendFactor::SrcAlphaSaturated => "src-alpha-saturated",
        api::BlendFactor::Src1 => "src1",
        api::BlendFactor::OneMinusSrc1 => "one-minus-src1",
    })
}

fn blend_operation(operation: api::BlendOperation) -> &'static str {
    match operation {
        api::BlendOperation::Add => "add",
        api::BlendOperation::Subtract => "subtract",
        api::BlendOperation::ReverseSubtract => "reverse-subtract",
        api::BlendOperation::Min => "min",
        api::BlendOperation::Max => "max",
    }
}

fn compare(value: api::CompareFunction) -> &'static str {
    match value {
        api::CompareFunction::Never => "never",
        api::CompareFunction::Less => "less",
        api::CompareFunction::Equal => "equal",
        api::CompareFunction::LessEqual => "less-equal",
        api::CompareFunction::Greater => "greater",
        api::CompareFunction::NotEqual => "not-equal",
        api::CompareFunction::GreaterEqual => "greater-equal",
        api::CompareFunction::Always => "always",
    }
}

fn vertex_format(value: api::VertexFormat) -> Result<&'static str> {
    Ok(match value {
        api::VertexFormat::Float32(1) => "float32",
        api::VertexFormat::Float32(2) => "float32x2",
        api::VertexFormat::Float32(3) => "float32x3",
        api::VertexFormat::Float32(4) => "float32x4",
        api::VertexFormat::Uint32(1) => "uint32",
        api::VertexFormat::Uint32(2) => "uint32x2",
        api::VertexFormat::Uint32(3) => "uint32x3",
        api::VertexFormat::Uint32(4) => "uint32x4",
        api::VertexFormat::Sint32(1) => "sint32",
        api::VertexFormat::Sint32(2) => "sint32x2",
        api::VertexFormat::Sint32(3) => "sint32x3",
        api::VertexFormat::Sint32(4) => "sint32x4",
        _ => return Err(Error::Unsupported("WebGPU vertex format")),
    })
}
