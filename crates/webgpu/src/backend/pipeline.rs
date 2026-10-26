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
    type NagaFuture<'a> = Pin<Box<dyn Future<Output = Result<JsValue>> + 'a>>;

    fn shader_route(_: &WebGpuDevice, desc: &api::ShaderDesc) -> api::ShaderRoute {
        shader_route(desc.format, super::shader_worker::is_configured())
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
            field(
                &native,
                "code",
                JsValue::from_str(if super::immediate_shader::has_immediate(source)? {
                    ""
                } else {
                    source
                }),
            )?;
            if let Some(label) = &desc.label {
                field(&native, "label", JsValue::from_str(label))?;
            }
            let module =
                js::call1(&device.0.device, "createShaderModule", &native).map_err(error)?;
            super::subgroup::remember(&module, &source);
            Ok(module)
        })();
        ready(result)
    }

    fn compile_naga<'a>(
        device: &'a WebGpuDevice,
        desc: &'a api::ShaderDesc,
        _: &'a api::ShaderTarget,
    ) -> Self::NagaFuture<'a> {
        let bytes = desc.bytes.to_vec();
        let label = desc.label.clone();
        Box::pin(async move {
            let source = super::shader_worker::translate_async(bytes).await?;
            let native = object();
            field(
                &native,
                "code",
                JsValue::from_str(if super::immediate_shader::has_immediate(&source)? {
                    ""
                } else {
                    &source
                }),
            )?;
            if let Some(label) = &label {
                field(&native, "label", JsValue::from_str(label))?;
            }
            let module =
                js::call1(&device.0.device, "createShaderModule", &native).map_err(error)?;
            super::subgroup::remember(&module, &source);
            Ok(module)
        })
    }
}

fn shader_route(format: api::ShaderFormat, spirv_worker_configured: bool) -> api::ShaderRoute {
    match format {
        api::ShaderFormat::Wgsl => api::ShaderRoute::Direct(api::ShaderTarget {
            format: api::ShaderFormat::Wgsl,
            environment: "browser-webgpu".into(),
            abi_key: "wgsl-group-binding".into(),
        }),
        api::ShaderFormat::SpirV if spirv_worker_configured => {
            api::ShaderRoute::Naga(api::ShaderTarget {
                format: api::ShaderFormat::Wgsl,
                environment: "browser-webgpu".into(),
                abi_key: "spirv-to-wgsl-group-binding".into(),
            })
        }
        _ => api::ShaderRoute::Unsupported,
    }
}

#[cfg(test)]
mod tests {
    use super::shader_route;
    use fluxel_rhi_core::api::{ShaderFormat, ShaderRoute};

    #[test]
    fn spirv_is_not_advertised_without_its_worker_asset() {
        assert!(matches!(
            shader_route(ShaderFormat::SpirV, false),
            ShaderRoute::Unsupported
        ));
        assert!(matches!(
            shader_route(ShaderFormat::SpirV, true),
            ShaderRoute::Naga(_)
        ));
    }
}

impl api::PipelineBackend for WebGpu {
    type RasterPipelineFuture<'a> = Pin<Box<dyn Future<Output = Result<JsValue>> + 'a>>;

    fn create_raster_pipeline<'a>(
        device: &'a WebGpuDevice,
        desc: &'a api::RasterPipelineDesc<Self>,
    ) -> Self::RasterPipelineFuture<'a> {
        Box::pin(async move {
            let native = raster_descriptor(device, desc)?;
            let promise =
                js::call1(&device.0.device, "createRenderPipelineAsync", &native).map_err(error)?;
            let pipeline = JsFuture::from(Promise::from(promise))
                .await
                .map_err(|error| Error::Shader(js::message(&error)))?;
            // SAFETY: Core checked layout ownership before entering this method.
            super::immediate::associate_descriptor(&pipeline, &native);
            super::immediate::remember_samples(&pipeline, desc.multisample.count);
            Ok(pipeline)
        })
    }
}

fn raster_descriptor(
    device: &WebGpuDevice,
    desc: &api::RasterPipelineDesc<WebGpu>,
) -> Result<JsValue> {
    if desc.multiview.is_some() || desc.primitive.conservative {
        return Err(Error::Unsupported("WebGPU raster pipeline option"));
    }
    if !matches!(desc.primitive.polygon_mode, api::PolygonMode::Fill) {
        return Err(Error::Unsupported("WebGPU polygon mode"));
    }
    if desc.primitive.unclipped_depth && !js::has_feature(&device.0.device, "depth-clip-control") {
        return Err(Error::Unsupported("WebGPU depth-clip-control feature"));
    }
    let native = object();
    // SAFETY: `Device::raster_pipeline` validates the layout, shader, and cache
    // DeviceIds against its device before it calls this backend future.
    let layout = unsafe { desc.layout.native_ref() }.clone();
    let fragment_module = desc.fragment.as_ref().map(|fragment| {
        // SAFETY: Core checked this shader's ownership.
        unsafe { fragment.module.native_ref() }.clone()
    });
    let immediate_info = super::immediate::raster_info(
        device,
        &layout,
        fragment_module
            .as_ref()
            .zip(desc.fragment.as_ref().map(|f| f.entry.as_str())),
    )?;
    super::immediate::remember_descriptor(&native, immediate_info);
    field(&native, "layout", layout)?;
    if let Some(label) = &desc.label {
        field(&native, "label", JsValue::from_str(label))?;
    }

    let vertex = object();
    // SAFETY: `Device::raster_pipeline` validated the vertex shader DeviceId;
    // the cloned JS module is held by the browser descriptor for this call.
    let vertex_module = unsafe { desc.vertex.module.native_ref() }.clone();
    let vertex_module = super::immediate::variant_module(
        device,
        &vertex_module,
        immediate_info[0],
        &desc.vertex.entry,
        None,
    )?;
    field(&vertex, "module", vertex_module)?;
    field(&vertex, "entryPoint", JsValue::from_str(&desc.vertex.entry))?;
    field(&vertex, "constants", constants(&desc.vertex.constants)?)?;
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
        // SAFETY: the same public pipeline validation checked this optional
        // fragment shader against the device that owns this backend call.
        let fragment_module = unsafe { fragment.module.native_ref() }.clone();
        let fragment_module = super::immediate::variant_module(
            device,
            &fragment_module,
            immediate_info[1],
            &fragment.entry,
            None,
        )?;
        field(&part, "module", fragment_module)?;
        field(&part, "entryPoint", JsValue::from_str(&fragment.entry))?;
        field(&part, "constants", constants(&fragment.constants)?)?;
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
    if let Some(format) = desc.primitive.strip_index_format {
        field(
            &primitive,
            "stripIndexFormat",
            JsValue::from_str(match format {
                api::IndexFormat::Uint16 => "uint16",
                api::IndexFormat::Uint32 => "uint32",
            }),
        )?;
    }
    if desc.primitive.unclipped_depth {
        field(&primitive, "unclippedDepth", JsValue::TRUE)?;
    }
    field(&native, "primitive", primitive)?;
    if let Some(depth) = &desc.depth_stencil {
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
        field(&state, "stencilFront", stencil_face(depth.stencil_front)?)?;
        field(&state, "stencilBack", stencil_face(depth.stencil_back)?)?;
        field(
            &state,
            "stencilReadMask",
            JsValue::from_f64(depth.stencil_read_mask as f64),
        )?;
        field(
            &state,
            "stencilWriteMask",
            JsValue::from_f64(depth.stencil_write_mask as f64),
        )?;
        field(&native, "depthStencil", state)?;
    }
    if desc.depth_stencil.is_none()
        && (desc.fragment.is_none() || desc.color_targets.iter().all(Option::is_none))
    {
        let state = object();
        field(&state, "format", JsValue::from_str("depth32float"))?;
        field(&state, "depthWriteEnabled", JsValue::FALSE)?;
        field(&state, "depthCompare", JsValue::from_str("always"))?;
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
        api::VertexFormat::Float16(1) => "float16",
        api::VertexFormat::Float16(2) => "float16x2",
        api::VertexFormat::Float16(4) => "float16x4",
        api::VertexFormat::Uint16(1) => "uint16",
        api::VertexFormat::Uint16(2) => "uint16x2",
        api::VertexFormat::Uint16(4) => "uint16x4",
        api::VertexFormat::Sint16(1) => "sint16",
        api::VertexFormat::Sint16(2) => "sint16x2",
        api::VertexFormat::Sint16(4) => "sint16x4",
        api::VertexFormat::Uint8(1) => "uint8",
        api::VertexFormat::Uint8(2) => "uint8x2",
        api::VertexFormat::Uint8(4) => "uint8x4",
        api::VertexFormat::Sint8(1) => "sint8",
        api::VertexFormat::Sint8(2) => "sint8x2",
        api::VertexFormat::Sint8(4) => "sint8x4",
        api::VertexFormat::Unorm8(1) => "unorm8",
        api::VertexFormat::Unorm8(2) => "unorm8x2",
        api::VertexFormat::Unorm8(4) => "unorm8x4",
        api::VertexFormat::Snorm8(1) => "snorm8",
        api::VertexFormat::Snorm8(2) => "snorm8x2",
        api::VertexFormat::Snorm8(4) => "snorm8x4",
        api::VertexFormat::Unorm16(1) => "unorm16",
        api::VertexFormat::Unorm16(2) => "unorm16x2",
        api::VertexFormat::Unorm16(4) => "unorm16x4",
        api::VertexFormat::Snorm16(1) => "snorm16",
        api::VertexFormat::Snorm16(2) => "snorm16x2",
        api::VertexFormat::Snorm16(4) => "snorm16x4",
        _ => return Err(Error::Unsupported("WebGPU vertex format")),
    })
}

fn stencil_face(face: api::StencilFaceState) -> Result<JsValue> {
    let state = object();
    field(&state, "compare", JsValue::from_str(compare(face.compare)))?;
    field(&state, "failOp", JsValue::from_str(stencil_op(face.fail)))?;
    field(
        &state,
        "depthFailOp",
        JsValue::from_str(stencil_op(face.depth_fail)),
    )?;
    field(&state, "passOp", JsValue::from_str(stencil_op(face.pass)))?;
    Ok(state)
}

fn stencil_op(op: api::StencilOperation) -> &'static str {
    match op {
        api::StencilOperation::Keep => "keep",
        api::StencilOperation::Zero => "zero",
        api::StencilOperation::Replace => "replace",
        api::StencilOperation::Invert => "invert",
        api::StencilOperation::IncrementClamp => "increment-clamp",
        api::StencilOperation::DecrementClamp => "decrement-clamp",
        api::StencilOperation::IncrementWrap => "increment-wrap",
        api::StencilOperation::DecrementWrap => "decrement-wrap",
    }
}

/// WebGPU uses JavaScript numbers for override constants, including booleans.
pub(super) fn constants(values: &[(String, api::SpecializationValue)]) -> Result<JsValue> {
    let result = object();
    for (name, value) in values {
        use api::SpecializationValue as V;
        let number = match *value {
            V::Bool(v) => u8::from(v) as f64,
            V::U8(v) => v as f64,
            V::I8(v) => v as f64,
            V::U16(v) => v as f64,
            V::I16(v) => v as f64,
            V::U32(v) => v as f64,
            V::I32(v) => v as f64,
            V::U64(v) if v <= (1_u64 << 53) => v as f64,
            V::I64(v) if v.unsigned_abs() <= (1_u64 << 53) => v as f64,
            V::U64(_) | V::I64(_) => {
                return Err(Error::Unsupported(
                    "WebGPU integer override exceeds JavaScript precision",
                ));
            }
            V::F16Bits(v) => half_to_f64(v),
            V::F32(v) => v as f64,
            V::F64(v) => v,
        };
        field(&result, name, JsValue::from_f64(number))?;
    }
    Ok(result)
}

fn half_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x3ff;
    match exponent {
        0 => sign * (mantissa as f64) * 2_f64.powi(-24),
        31 if mantissa == 0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1.0 + (mantissa as f64) / 1024.0) * 2_f64.powi(exponent as i32 - 15),
    }
}
