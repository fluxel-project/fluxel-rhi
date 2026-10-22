//! Depth blits use fragment depth writes to preserve the rest of the attachment.
use super::{
    Error, FilterMode, JsValue, Result, WebGpu, WebGpuEncoder, api, call, call0, call1, field,
    mip_extent, number, object,
};
use js_sys::Array;

pub(super) fn blit_depth(
    encoder: &mut WebGpuEncoder,
    src: api::TextureRegion<'_, WebGpu>,
    dst: api::TextureRegion<'_, WebGpu>,
    filter: FilterMode,
) -> Result<()> {
    // SAFETY: the public encoder checked that both textures belong to its device.
    let source = unsafe { src.texture.native_ref() };
    let target = unsafe { dst.texture.native_ref() };
    let sd = &source.desc;
    let td = &target.desc;
    if sd.dimension != api::TextureDimension::D2
        || td.dimension != api::TextureDimension::D2
        || sd.samples != td.samples
        || (sd.samples > 1 && filter != FilterMode::Nearest)
        || src.aspect != api::TextureAspects::DEPTH
        || dst.aspect != api::TextureAspects::DEPTH
        || !sd.usage.contains(api::TextureUsage::COPY_SRC)
        || !td.usage.contains(api::TextureUsage::COPY_DST)
        || src.extent[2] != 1
        || dst.extent[2] != 1
        || src.extent.contains(&0)
        || dst.extent.contains(&0)
        || src.mip >= sd.mip_levels
        || dst.mip >= td.mip_levels
    {
        return Err(Error::Unsupported("WebGPU depth blit descriptor"));
    }
    let sw = mip_extent(sd.extent[0], src.mip);
    let sh = mip_extent(sd.extent[1], src.mip);
    let dw = mip_extent(td.extent[0], dst.mip);
    let dh = mip_extent(td.extent[1], dst.mip);
    if src.origin[0]
        .checked_add(src.extent[0])
        .is_none_or(|v| v > sw)
        || src.origin[1]
            .checked_add(src.extent[1])
            .is_none_or(|v| v > sh)
        || dst.origin[0]
            .checked_add(dst.extent[0])
            .is_none_or(|v| v > dw)
        || dst.origin[1]
            .checked_add(dst.extent[1])
            .is_none_or(|v| v > dh)
        || src.origin[2] >= sd.array_layers
        || dst.origin[2] >= td.array_layers
    {
        return Err(Error::InvalidRange);
    }
    let fmt = super::super::resource::texture_format(td.format)?;
    let sf = super::super::resource::texture_format(sd.format)?;
    if !fmt.starts_with("depth") || !sf.starts_with("depth") {
        return Err(Error::Unsupported("WebGPU depth blit format"));
    }
    let uv = format!(
        "vec2<f32>({:.9}, {:.9}) + uv * vec2<f32>({:.9}, {:.9})",
        src.origin[0] as f32 / sw as f32,
        src.origin[1] as f32 / sh as f32,
        src.extent[0] as f32 / sw as f32,
        src.extent[1] as f32 / sh as f32
    );
    let value = match filter {
        FilterMode::Nearest => {
            "textureLoad(tex, vec2<i32>(min(vec2<u32>(uv * vec2<f32>(textureDimensions(tex))), textureDimensions(tex) - vec2<u32>(1u))), 0)"
        }
        FilterMode::Linear => "linear_depth(uv)",
    };
    let mut code = format!(
        "@group(0) @binding(0) var tex: texture_depth_2d; fn linear_depth(uv: vec2<f32>) -> f32 {{ let size=vec2<i32>(textureDimensions(tex)); let p=uv*vec2<f32>(size)-vec2<f32>(0.5); let base=vec2<i32>(floor(p)); let f=fract(p); let a=clamp(base,vec2<i32>(0),size-vec2<i32>(1)); let b=clamp(base+vec2<i32>(1),vec2<i32>(0),size-vec2<i32>(1)); return mix(mix(textureLoad(tex,a,0),textureLoad(tex,vec2<i32>(b.x,a.y),0),f.x),mix(textureLoad(tex,vec2<i32>(a.x,b.y),0),textureLoad(tex,b,0),f.x),f.y); }} struct O {{ @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32>, }}; @vertex fn vs(@builtin(vertex_index) i: u32) -> O {{ let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)); var o: O; o.pos = vec4<f32>(uv * vec2<f32>(2., -2.) + vec2<f32>(-1., 1.), 0., 1.); o.uv = {uv}; return o; }} @fragment fn fs(o: O) -> @builtin(frag_depth) f32 {{ let uv = o.uv; return {value}; }}"
    );
    if sd.samples > 1 {
        code = code
            .replace("texture_depth_2d", "texture_depth_multisampled_2d")
            .replace(
                "fn fs(o: O)",
                "fn fs(o: O, @builtin(sample_index) sample: u32)",
            )
            .replace(
                "textureDimensions(tex) - vec2<u32>(1u))), 0)",
                "textureDimensions(tex) - vec2<u32>(1u))), i32(sample))",
            );
    }
    let md = object();
    field(&md, "code", JsValue::from_str(&code))?;
    let module = call1(&encoder.device.0.device, "createShaderModule", &md)?;
    let vertex = object();
    field(&vertex, "module", module.clone())?;
    field(&vertex, "entryPoint", JsValue::from_str("vs"))?;
    let fragment = object();
    field(&fragment, "module", module)?;
    field(&fragment, "entryPoint", JsValue::from_str("fs"))?;
    field(&fragment, "targets", Array::new().into())?;
    let ds = object();
    field(&ds, "format", JsValue::from_str(fmt))?;
    field(&ds, "depthWriteEnabled", JsValue::TRUE)?;
    field(&ds, "depthCompare", JsValue::from_str("always"))?;
    let pd = object();
    field(&pd, "layout", JsValue::from_str("auto"))?;
    field(&pd, "vertex", vertex)?;
    field(&pd, "fragment", fragment)?;
    field(&pd, "depthStencil", ds)?;
    let ms = object();
    field(&ms, "count", number(sd.samples))?;
    field(&pd, "multisample", ms)?;
    let pipeline = call1(&encoder.device.0.device, "createRenderPipeline", &pd)?;
    let layout = call1(&pipeline, "getBindGroupLayout", &number(0))?;
    let alias = js_sys::Object::is(&source.raw, &target.raw);
    let sampled = if alias {
        let desc = object();
        let size = object();
        field(&size, "width", number(sw))?;
        field(&size, "height", number(sh))?;
        field(&size, "depthOrArrayLayers", number(1))?;
        field(&desc, "size", size)?;
        field(&desc, "sampleCount", number(sd.samples))?;
        field(&desc, "format", JsValue::from_str(sf))?;
        field(&desc, "usage", number(2 | 4 | 16))?;
        let temp = call1(&encoder.device.0.device, "createTexture", &desc)?;
        let from = object();
        field(&from, "texture", source.raw.clone())?;
        field(&from, "mipLevel", number(src.mip))?;
        let origin = object();
        field(&origin, "z", number(src.origin[2]))?;
        field(&from, "origin", origin)?;
        let to = object();
        field(&to, "texture", temp.clone())?;
        let extent = object();
        field(&extent, "width", number(sw))?;
        field(&extent, "height", number(sh))?;
        field(&extent, "depthOrArrayLayers", number(1))?;
        call(&encoder.raw, "copyTextureToTexture", &[from, to, extent])?;
        temp
    } else {
        source.raw.clone()
    };
    let vd = object();
    field(&vd, "dimension", JsValue::from_str("2d"))?;
    field(&vd, "aspect", JsValue::from_str("depth-only"))?;
    field(&vd, "baseMipLevel", number(if alias { 0 } else { src.mip }))?;
    field(&vd, "mipLevelCount", number(1))?;
    field(
        &vd,
        "baseArrayLayer",
        number(if alias { 0 } else { src.origin[2] }),
    )?;
    field(&vd, "arrayLayerCount", number(1))?;
    let sv = call1(&sampled, "createView", &vd)?;
    let vd = object();
    field(&vd, "dimension", JsValue::from_str("2d"))?;
    field(&vd, "baseMipLevel", number(dst.mip))?;
    field(&vd, "mipLevelCount", number(1))?;
    field(&vd, "baseArrayLayer", number(dst.origin[2]))?;
    field(&vd, "arrayLayerCount", number(1))?;
    let dv = call1(&target.raw, "createView", &vd)?;
    let entries = Array::new();
    let e = object();
    field(&e, "binding", number(0))?;
    field(&e, "resource", sv.clone())?;
    entries.push(&e);
    let bd = object();
    field(&bd, "layout", layout)?;
    field(&bd, "entries", entries.into())?;
    let bind = call1(&encoder.device.0.device, "createBindGroup", &bd)?;
    let a = object();
    field(&a, "view", dv.clone())?;
    field(&a, "depthLoadOp", JsValue::from_str("load"))?;
    field(&a, "depthStoreOp", JsValue::from_str("store"))?;
    if fmt.contains("stencil") {
        field(&a, "stencilReadOnly", JsValue::TRUE)?;
    }
    let pass_desc = object();
    field(&pass_desc, "colorAttachments", Array::new().into())?;
    field(&pass_desc, "depthStencilAttachment", a)?;
    let pass = call1(&encoder.raw, "beginRenderPass", &pass_desc)?;
    call1(&pass, "setPipeline", &pipeline)?;
    call(&pass, "setBindGroup", &[number(0), bind.clone()])?;
    call(
        &pass,
        "setViewport",
        &[
            number(dst.origin[0]),
            number(dst.origin[1]),
            number(dst.extent[0]),
            number(dst.extent[1]),
            number(0),
            number(1),
        ],
    )?;
    call(
        &pass,
        "setScissorRect",
        &[
            number(dst.origin[0]),
            number(dst.origin[1]),
            number(dst.extent[0]),
            number(dst.extent[1]),
        ],
    )?;
    call(&pass, "draw", &[number(3), number(1), number(0), number(0)])?;
    call0(&pass, "end")?;
    encoder.keep_alive.extend([
        source.raw.clone(),
        target.raw.clone(),
        sampled,
        sv,
        dv,
        bind,
        pipeline,
    ]);
    Ok(())
}
