//! Native render-pass lowering for a color mip blit. WebGPU has no scaled copy.

use fluxel_rhi_core::api::{self, Error, FilterMode, Result};
use js_sys::{Array, Object};
use wasm_bindgen::JsValue;

use super::{command::WebGpuEncoder, instance::WebGpu, js};

const SHADER: &str = r#"
@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VertexOut;
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(source_texture, source_sampler, in.uv);
}
"#;

pub(super) struct BlitCache {
    pipeline: JsValue,
    layout: JsValue,
    linear: JsValue,
    nearest: JsValue,
}

fn object() -> JsValue {
    Object::new().into()
}

fn field(object: &JsValue, name: &str, value: JsValue) -> Result<()> {
    js::set(object, name, &value).map_err(|error| Error::Backend(js::message(&error)))
}

fn call(receiver: &JsValue, name: &str, args: &[JsValue]) -> Result<JsValue> {
    js::call(receiver, name, args).map_err(|error| Error::Backend(js::message(&error)))
}

fn call1(receiver: &JsValue, name: &str, arg: &JsValue) -> Result<JsValue> {
    js::call1(receiver, name, arg).map_err(|error| Error::Backend(js::message(&error)))
}

fn call0(receiver: &JsValue, name: &str) -> Result<JsValue> {
    js::call0(receiver, name).map_err(|error| Error::Backend(js::message(&error)))
}

fn number(value: u32) -> JsValue {
    JsValue::from_f64(value as f64)
}

fn cache(device: &JsValue) -> Result<BlitCache> {
    let module_desc = object();
    field(&module_desc, "code", JsValue::from_str(SHADER))?;
    let module = call1(device, "createShaderModule", &module_desc)?;
    let vertex = object();
    field(&vertex, "module", module.clone())?;
    field(&vertex, "entryPoint", JsValue::from_str("vs_main"))?;
    let fragment = object();
    field(&fragment, "module", module)?;
    field(&fragment, "entryPoint", JsValue::from_str("fs_main"))?;
    let target = object();
    field(&target, "format", JsValue::from_str("rgba8unorm"))?;
    let targets = Array::new();
    targets.push(&target);
    field(&fragment, "targets", targets.into())?;
    let desc = object();
    field(&desc, "layout", JsValue::from_str("auto"))?;
    field(&desc, "vertex", vertex)?;
    field(&desc, "fragment", fragment)?;
    let pipeline = call1(device, "createRenderPipeline", &desc)?;
    let layout = call1(&pipeline, "getBindGroupLayout", &number(0))?;
    let make_sampler = |mode: &str| -> Result<JsValue> {
        let desc = object();
        field(&desc, "minFilter", JsValue::from_str(mode))?;
        field(&desc, "magFilter", JsValue::from_str(mode))?;
        call1(device, "createSampler", &desc)
    };
    Ok(BlitCache {
        pipeline,
        layout,
        linear: make_sampler("linear")?,
        nearest: make_sampler("nearest")?,
    })
}

fn mip_extent(base: u32, mip: u32) -> u32 {
    base.checked_shr(mip).unwrap_or(0).max(1)
}

pub(super) fn blit(
    encoder: &mut WebGpuEncoder,
    src: api::TextureRegion<'_, WebGpu>,
    dst: api::TextureRegion<'_, WebGpu>,
    filter: FilterMode,
) -> Result<()> {
    let source = unsafe { src.texture.native_ref() };
    let target = unsafe { dst.texture.native_ref() };
    let desc = &source.desc;
    if encoder.pass.is_some() {
        return Err(Error::InvalidState);
    }
    if desc.dimension != api::TextureDimension::D2
        || desc.format != api::TextureFormat::Rgba8Unorm
        || desc.samples != 1
        || target.desc.format != desc.format
        || target.desc.samples != 1
        || src.aspect != api::TextureAspects::COLOR
        || dst.aspect != api::TextureAspects::COLOR
        || src.origin != [0, 0, 0]
        || dst.origin != [0, 0, 0]
        || src.extent
            != [
                mip_extent(desc.extent[0], src.mip),
                mip_extent(desc.extent[1], src.mip),
                1,
            ]
        || dst.extent
            != [
                mip_extent(target.desc.extent[0], dst.mip),
                mip_extent(target.desc.extent[1], dst.mip),
                1,
            ]
        || src.mip >= desc.mip_levels
        || dst.mip >= target.desc.mip_levels
        || !desc.usage.contains(api::TextureUsage::TEXTURE_BINDING)
        || !target.desc.usage.contains(api::TextureUsage::COPY_DST)
    {
        return Err(Error::Unsupported("WebGPU full RGBA8 mip blit"));
    }
    if let Some(message) = encoder.device.0.validation_error.borrow().clone() {
        return Err(Error::Backend(message));
    }
    if encoder.device.0.blit_cache.borrow().is_none() {
        let new_cache = cache(&encoder.device.0.device)?;
        *encoder.device.0.blit_cache.borrow_mut() = Some(new_cache);
    }
    let cache_ref = encoder.device.0.blit_cache.borrow();
    let cache = cache_ref.as_ref().ok_or(Error::InvalidState)?;
    let source_view_desc = object();
    field(&source_view_desc, "dimension", JsValue::from_str("2d"))?;
    field(&source_view_desc, "baseMipLevel", number(src.mip))?;
    field(&source_view_desc, "mipLevelCount", number(1))?;
    field(&source_view_desc, "baseArrayLayer", number(0))?;
    field(&source_view_desc, "arrayLayerCount", number(1))?;
    let source_view = call1(&source.raw, "createView", &source_view_desc)?;

    // The public blit contract only requires COPY_DST on the destination.
    // Render into a private attachment, then copy the pixels into that mip.
    let temp_desc = object();
    let size = object();
    field(&size, "width", number(dst.extent[0]))?;
    field(&size, "height", number(dst.extent[1]))?;
    field(&size, "depthOrArrayLayers", number(1))?;
    field(&temp_desc, "size", size)?;
    field(&temp_desc, "format", JsValue::from_str("rgba8unorm"))?;
    field(&temp_desc, "usage", number(17))?; // COPY_SRC | RENDER_ATTACHMENT
    let temp = call1(&encoder.device.0.device, "createTexture", &temp_desc)?;
    let temp_view = call0(&temp, "createView")?;
    let bind_desc = object();
    field(&bind_desc, "layout", cache.layout.clone())?;
    let entries = Array::new();
    for (binding, resource) in [
        (0, source_view.clone()),
        (
            1,
            match filter {
                FilterMode::Linear => cache.linear.clone(),
                FilterMode::Nearest => cache.nearest.clone(),
            },
        ),
    ] {
        let entry = object();
        field(&entry, "binding", number(binding))?;
        field(&entry, "resource", resource)?;
        entries.push(&entry);
    }
    field(&bind_desc, "entries", entries.into())?;
    let bind = call1(&encoder.device.0.device, "createBindGroup", &bind_desc)?;
    let pass_desc = object();
    let attachment = object();
    field(&attachment, "view", temp_view.clone())?;
    field(&attachment, "loadOp", JsValue::from_str("clear"))?;
    field(&attachment, "storeOp", JsValue::from_str("store"))?;
    let clear = object();
    for name in ["r", "g", "b", "a"] {
        field(&clear, name, number(0))?;
    }
    field(&attachment, "clearValue", clear)?;
    let colors = Array::new();
    colors.push(&attachment);
    field(&pass_desc, "colorAttachments", colors.into())?;
    let pass = call1(&encoder.raw, "beginRenderPass", &pass_desc)?;
    call1(&pass, "setPipeline", &cache.pipeline)?;
    call(&pass, "setBindGroup", &[number(0), bind.clone()])?;
    call(&pass, "draw", &[number(3), number(1), number(0), number(0)])?;
    call0(&pass, "end")?;
    let source_copy = object();
    field(&source_copy, "texture", temp.clone())?;
    let target_copy = object();
    field(&target_copy, "texture", target.raw.clone())?;
    field(&target_copy, "mipLevel", number(dst.mip))?;
    let extent = object();
    field(&extent, "width", number(dst.extent[0]))?;
    field(&extent, "height", number(dst.extent[1]))?;
    field(&extent, "depthOrArrayLayers", number(1))?;
    call(
        &encoder.raw,
        "copyTextureToTexture",
        &[source_copy, target_copy, extent],
    )?;
    encoder.keep_alive.extend([
        source.raw.clone(),
        target.raw.clone(),
        temp,
        source_view,
        temp_view,
        bind,
        cache.pipeline.clone(),
    ]);
    Ok(())
}
