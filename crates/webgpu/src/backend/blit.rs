//! Native render-pass lowering for a color mip blit. WebGPU has no scaled copy.
//!
//! The device-local cache contains only immutable helper pipeline objects. Each
//! invocation records its transient views, bind group, and texture in the live
//! native encoder, which retains them through queue completion.

use fluxel_rhi_core::api::{self, Error, FilterMode, Result};
use js_sys::{Array, Object};
use wasm_bindgen::JsValue;

use super::{command::WebGpuEncoder, instance::WebGpu, js};

#[path = "blit_depth.rs"]
mod depth;

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

fn cache(
    device: &JsValue,
    target_format: &str,
    uv_scale: [f32; 2],
    uv_offset: [f32; 2],
    sample_type: &str,
    use_load: bool,
    source_z: Option<(u32, f32)>,
    source_1d: bool,
) -> Result<BlitCache> {
    let module_desc = object();
    let mut source = SHADER.replace(
        "out.uv = uv;",
        &format!(
            "out.uv = uv * vec2<f32>({:.9}, {:.9}) + vec2<f32>({:.9}, {:.9});",
            uv_scale[0], uv_scale[1], uv_offset[0], uv_offset[1]
        ),
    );
    if use_load {
        source = source.replace(
            "var source_texture: texture_2d<f32>;",
            &format!("var source_texture: texture_2d<{sample_type}>;"),
        );
        source = source.replace("@group(0) @binding(1) var source_sampler: sampler;", "");
        source = source.replace("@location(0) vec4<f32> {\n    return textureSample(source_texture, source_sampler, in.uv);", &format!("@location(0) vec4<{sample_type}> {{\n    let size = textureDimensions(source_texture);\n    let coord = min(vec2<u32>(in.uv * vec2<f32>(size)), size - vec2<u32>(1u));\n    return textureLoad(source_texture, vec2<i32>(coord), 0);"));
    }
    if let Some((slice, normalized_z)) = source_z {
        source = source.replace("texture_2d<", "texture_3d<");
        source = source.replace(
            "let size = textureDimensions(source_texture);",
            "let size = textureDimensions(source_texture).xy;",
        );
        source = source.replace(
            "textureLoad(source_texture, vec2<i32>(coord), 0)",
            &format!("textureLoad(source_texture, vec3<i32>(vec2<i32>(coord), {slice}), 0)"),
        );
        source = source.replace(
            "textureSample(source_texture, source_sampler, in.uv)",
            &format!(
                "textureSample(source_texture, source_sampler, vec3<f32>(in.uv, {:.9}))",
                normalized_z
            ),
        );
    }
    if source_1d {
        source = source.replace("texture_2d<", "texture_1d<");
        source = source.replace(
            "let size = textureDimensions(source_texture);",
            "let size = vec2<u32>(textureDimensions(source_texture), 1u);",
        );
        source = source.replace(
            "textureLoad(source_texture, vec2<i32>(coord), 0)",
            "textureLoad(source_texture, i32(coord.x), 0)",
        );
        source = source.replace(
            "textureSample(source_texture, source_sampler, in.uv)",
            "textureSample(source_texture, source_sampler, in.uv.x)",
        );
    }
    field(&module_desc, "code", JsValue::from_str(&source))?;
    let module = call1(device, "createShaderModule", &module_desc)?;
    let vertex = object();
    field(&vertex, "module", module.clone())?;
    field(&vertex, "entryPoint", JsValue::from_str("vs_main"))?;
    let fragment = object();
    field(&fragment, "module", module)?;
    field(&fragment, "entryPoint", JsValue::from_str("fs_main"))?;
    let target = object();
    field(&target, "format", JsValue::from_str(target_format))?;
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
    blit_inner(encoder, src, dst, filter, None)
}

fn blit_inner(
    encoder: &mut WebGpuEncoder,
    src: api::TextureRegion<'_, WebGpu>,
    dst: api::TextureRegion<'_, WebGpu>,
    filter: FilterMode,
    source_z: Option<f32>,
) -> Result<()> {
    // SAFETY: `CommandEncoder::blit` checked both region DeviceIds against this
    // encoder; their raw JS objects are retained with the finished command buffer.
    let source = unsafe { src.texture.native_ref() };
    let target = unsafe { dst.texture.native_ref() };
    let desc = &source.desc;
    if encoder.pass.is_some() {
        return Err(Error::InvalidState);
    }
    let source_layers = if desc.dimension == api::TextureDimension::D3 {
        mip_extent(desc.extent[2], src.mip)
    } else {
        desc.array_layers
    };
    let target_layers = if target.desc.dimension == api::TextureDimension::D3 {
        mip_extent(target.desc.extent[2], dst.mip)
    } else {
        target.desc.array_layers
    };
    if src.origin[2]
        .checked_add(src.extent[2])
        .is_none_or(|end| end > source_layers)
        || dst.origin[2]
            .checked_add(dst.extent[2])
            .is_none_or(|end| end > target_layers)
    {
        return Err(Error::InvalidRange);
    }
    if (src.extent[2] != dst.extent[2]
        && (desc.dimension != api::TextureDimension::D3
            || target.desc.dimension != api::TextureDimension::D3))
        || src.extent.contains(&0)
        || dst.extent.contains(&0)
    {
        return Err(Error::InvalidRange);
    }
    if src.extent[2] > 1 || dst.extent[2] > 1 {
        for layer in 0..dst.extent[2] {
            let z = src.origin[2] as f32
                + (layer as f32 + 0.5) * src.extent[2] as f32 / dst.extent[2] as f32;
            let source_layer = if desc.dimension == api::TextureDimension::D3 {
                (z.floor() as u32).min(
                    src.origin[2]
                        .checked_add(src.extent[2])
                        .ok_or(Error::InvalidRange)?
                        - 1,
                )
            } else {
                src.origin[2]
                    .checked_add(layer)
                    .ok_or(Error::InvalidRange)?
            };
            blit_inner(
                encoder,
                api::TextureRegion {
                    texture: src.texture,
                    mip: src.mip,
                    origin: [src.origin[0], src.origin[1], source_layer],
                    extent: [src.extent[0], src.extent[1], 1],
                    aspect: src.aspect,
                },
                api::TextureRegion {
                    texture: dst.texture,
                    mip: dst.mip,
                    origin: [
                        dst.origin[0],
                        dst.origin[1],
                        dst.origin[2]
                            .checked_add(layer)
                            .ok_or(Error::InvalidRange)?,
                    ],
                    extent: [dst.extent[0], dst.extent[1], 1],
                    aspect: dst.aspect,
                },
                filter,
                if desc.dimension == api::TextureDimension::D3 {
                    Some(z / mip_extent(desc.extent[2], src.mip) as f32)
                } else {
                    None
                },
            )?;
        }
        return Ok(());
    }
    if src.aspect == api::TextureAspects::DEPTH && dst.aspect == api::TextureAspects::DEPTH {
        return depth::blit_depth(encoder, src, dst, filter);
    }
    let (source_filterable, _, _, _, _) =
        super::resource_validation::capabilities(&encoder.device, desc.format);
    let (_, _, _, target_renderable, _) =
        super::resource_validation::capabilities(&encoder.device, target.desc.format);
    let target_name = super::resource::texture_format(target.desc.format)?;
    let source_name = super::resource::texture_format(desc.format)?;
    let sample_type = |name: &str| {
        if name.ends_with("uint") {
            "u32"
        } else if name.ends_with("sint") {
            "i32"
        } else {
            "f32"
        }
    };
    let source_type = sample_type(source_name);
    let target_type = sample_type(target_name);
    let use_load = filter == FilterMode::Nearest && (!source_filterable || source_type != "f32");
    if (filter == FilterMode::Linear && (!source_filterable || source_type != "f32"))
        || source_type != target_type
        || !target_renderable
        || target_name.starts_with("depth")
        || target_name == "stencil8"
        || desc.samples != 1
        || target.desc.samples != 1
        || src.aspect != api::TextureAspects::COLOR
        || dst.aspect != api::TextureAspects::COLOR
        || src.mip >= desc.mip_levels
        || dst.mip >= target.desc.mip_levels
        || !desc.usage.contains(api::TextureUsage::COPY_SRC)
        || !target.desc.usage.contains(api::TextureUsage::COPY_DST)
    {
        return Err(Error::Unsupported(
            "WebGPU sampled color blit format or usage",
        ));
    }
    if let Some(message) = encoder.device.0.validation_error.borrow().clone() {
        return Err(Error::Backend(message));
    }
    let source_width = mip_extent(desc.extent[0], src.mip);
    let source_height = mip_extent(desc.extent[1], src.mip);
    let target_width = mip_extent(target.desc.extent[0], dst.mip);
    let target_height = mip_extent(target.desc.extent[1], dst.mip);
    if src.origin[0]
        .checked_add(src.extent[0])
        .is_none_or(|end| end > source_width)
        || src.origin[1]
            .checked_add(src.extent[1])
            .is_none_or(|end| end > source_height)
        || dst.origin[0]
            .checked_add(dst.extent[0])
            .is_none_or(|end| end > target_width)
        || dst.origin[1]
            .checked_add(dst.extent[1])
            .is_none_or(|end| end > target_height)
        || src.origin[2]
            >= if desc.dimension == api::TextureDimension::D3 {
                mip_extent(desc.extent[2], src.mip)
            } else {
                desc.array_layers
            }
        || dst.origin[2]
            >= if target.desc.dimension == api::TextureDimension::D3 {
                mip_extent(target.desc.extent[2], dst.mip)
            } else {
                target.desc.array_layers
            }
    {
        return Err(Error::InvalidRange);
    }
    let target_format = super::resource::texture_format(target.desc.format)?;
    let cache = cache(
        &encoder.device.0.device,
        target_format,
        [
            src.extent[0] as f32 / source_width as f32,
            src.extent[1] as f32 / source_height as f32,
        ],
        [
            src.origin[0] as f32 / source_width as f32,
            src.origin[1] as f32 / source_height as f32,
        ],
        source_type,
        use_load,
        if desc.dimension == api::TextureDimension::D3 {
            Some((
                src.origin[2],
                source_z.unwrap_or(
                    (src.origin[2] as f32 + 0.5) / mip_extent(desc.extent[2], src.mip) as f32,
                ),
            ))
        } else {
            None
        },
        desc.dimension == api::TextureDimension::D1,
    )?;
    // A sampled intermediate supplies the private usage required by the
    // render implementation without changing the public COPY_SRC contract.
    let sampled = if desc.usage.contains(api::TextureUsage::TEXTURE_BINDING)
        || desc.usage.contains(api::TextureUsage::COPY_SRC)
    {
        source.raw.clone()
    } else {
        let intermediate_desc = object();
        let size = object();
        field(&size, "width", number(source_width))?;
        field(&size, "height", number(source_height))?;
        field(&size, "depthOrArrayLayers", number(1))?;
        field(&intermediate_desc, "size", size)?;
        field(
            &intermediate_desc,
            "format",
            JsValue::from_str(super::resource::texture_format(desc.format)?),
        )?;
        field(&intermediate_desc, "usage", number(2 | 4))?;
        let texture = call1(
            &encoder.device.0.device,
            "createTexture",
            &intermediate_desc,
        )?;
        let copy_src = object();
        field(&copy_src, "texture", source.raw.clone())?;
        field(&copy_src, "mipLevel", number(src.mip))?;
        let origin = object();
        field(&origin, "z", number(src.origin[2]))?;
        field(&copy_src, "origin", origin)?;
        let copy_dst = object();
        field(&copy_dst, "texture", texture.clone())?;
        let extent = object();
        field(&extent, "width", number(source_width))?;
        field(&extent, "height", number(source_height))?;
        field(&extent, "depthOrArrayLayers", number(1))?;
        call(
            &encoder.raw,
            "copyTextureToTexture",
            &[copy_src, copy_dst, extent],
        )?;
        texture
    };
    let direct = desc.usage.contains(api::TextureUsage::TEXTURE_BINDING)
        || desc.usage.contains(api::TextureUsage::COPY_SRC);
    let source_view_desc = object();
    field(
        &source_view_desc,
        "dimension",
        JsValue::from_str(match desc.dimension {
            api::TextureDimension::D1 => "1d",
            api::TextureDimension::D2 => "2d",
            api::TextureDimension::D3 => "3d",
        }),
    )?;
    field(
        &source_view_desc,
        "baseMipLevel",
        number(if direct { src.mip } else { 0 }),
    )?;
    field(&source_view_desc, "mipLevelCount", number(1))?;
    if desc.dimension == api::TextureDimension::D2 {
        field(
            &source_view_desc,
            "baseArrayLayer",
            number(if direct { src.origin[2] } else { 0 }),
        )?;
        field(&source_view_desc, "arrayLayerCount", number(1))?;
    }
    let source_view = call1(&sampled, "createView", &source_view_desc)?;

    // The public blit contract only requires COPY_DST on the destination.
    // Render into a private attachment, then copy the pixels into that mip.
    let temp_desc = object();
    let size = object();
    field(&size, "width", number(dst.extent[0]))?;
    field(&size, "height", number(dst.extent[1]))?;
    field(&size, "depthOrArrayLayers", number(1))?;
    field(&temp_desc, "size", size)?;
    field(&temp_desc, "format", JsValue::from_str(target_format))?;
    field(&temp_desc, "usage", number(17))?; // COPY_SRC | RENDER_ATTACHMENT
    let temp = call1(&encoder.device.0.device, "createTexture", &temp_desc)?;
    let temp_view = call0(&temp, "createView")?;
    let bind_desc = object();
    field(&bind_desc, "layout", cache.layout.clone())?;
    let entries = Array::new();
    let mut resources = vec![(0, source_view.clone())];
    if !use_load {
        resources.push((
            1,
            match filter {
                FilterMode::Linear => cache.linear.clone(),
                FilterMode::Nearest => cache.nearest.clone(),
            },
        ));
    }
    for (binding, resource) in resources {
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
    let origin = object();
    field(&origin, "x", number(dst.origin[0]))?;
    field(&origin, "y", number(dst.origin[1]))?;
    field(&origin, "z", number(dst.origin[2]))?;
    field(&target_copy, "origin", origin)?;
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
        sampled,
        target.raw.clone(),
        temp,
        source_view,
        temp_view,
        bind,
        cache.pipeline.clone(),
    ]);
    Ok(())
}
