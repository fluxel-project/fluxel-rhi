//! Copy a stencil aspect through a uint sampled view and stencil replacement.
use super::super::utility;
use super::{Result, WebGpu, WebGpuEncoder, api, error, field, js, number, object};
use js_sys::Array;
use wasm_bindgen::JsValue;

pub(super) fn copy(
    encoder: &mut WebGpuEncoder,
    src: api::TextureRegion<'_, WebGpu>,
    dst: api::TextureRegion<'_, WebGpu>,
) -> Result<()> {
    // SAFETY: command entry point validated device identity and texture regions.
    let source = unsafe { src.texture.native_ref() };
    let target = unsafe { dst.texture.native_ref() };
    let fmt = super::super::resource::texture_format(target.desc.format)?;
    let mut sample = "textureLoad(tex,vec2<i32>(p.xy),0).x".to_owned();
    let mut texture = "texture_2d<u32>".to_owned();
    if source.desc.samples > 1 {
        texture = "texture_multisampled_2d<u32>".to_owned();
        sample = "textureLoad(tex,vec2<i32>(p.xy),i32(sample)).x".to_owned();
    }
    let cache_device = encoder.device.clone();
    let make_pipeline = |key: String, code: String, write_mask: u32| -> Result<JsValue> {
        let native_device = cache_device.clone();
        utility::render_pipeline(&cache_device, key, move || {
            let md = object();
            field(&md, "code", JsValue::from_str(&code))?;
            let module =
                js::call1(&native_device.0.device, "createShaderModule", &md).map_err(error)?;
            let vertex = object();
            field(&vertex, "module", module.clone())?;
            field(&vertex, "entryPoint", JsValue::from_str("vs"))?;
            let fragment = object();
            field(&fragment, "module", module)?;
            field(&fragment, "entryPoint", JsValue::from_str("fs"))?;
            field(&fragment, "targets", Array::new().into())?;
            let stencil = object();
            field(&stencil, "compare", JsValue::from_str("always"))?;
            field(&stencil, "passOp", JsValue::from_str("replace"))?;
            let ds = object();
            field(&ds, "format", JsValue::from_str(fmt))?;
            field(&ds, "depthWriteEnabled", JsValue::FALSE)?;
            field(&ds, "depthCompare", JsValue::from_str("always"))?;
            field(&ds, "stencilFront", stencil.clone())?;
            field(&ds, "stencilBack", stencil)?;
            field(&ds, "stencilReadMask", number(255u32))?;
            field(&ds, "stencilWriteMask", number(write_mask))?;
            let pd = object();
            field(&pd, "layout", JsValue::from_str("auto"))?;
            field(&pd, "vertex", vertex)?;
            field(&pd, "fragment", fragment)?;
            field(&pd, "depthStencil", ds)?;
            let ms = object();
            field(&ms, "count", number(source.desc.samples))?;
            field(&pd, "multisample", ms)?;
            js::call1(&native_device.0.device, "createRenderPipeline", &pd).map_err(error)
        })
    };
    let vertex = "@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32>{let p=vec2<f32>(f32((i<<1u)&2u),f32(i&2u));return vec4<f32>(p*vec2<f32>(2.,-2.)+vec2<f32>(-1.,1.),0.,1.);}";
    let clear = make_pipeline(
        format!("stencil-copy-clear:{fmt}:{}", source.desc.samples),
        format!("{vertex}@fragment fn fs(){{}}"),
        255,
    )?;
    let mut planes = Vec::with_capacity(8);
    for bit in 0..8u32 {
        let mask = 1u32 << bit;
        let fragment_args = if source.desc.samples > 1 {
            ", @builtin(sample_index) sample:u32"
        } else {
            ""
        };
        let code = format!(
            "@group(0) @binding(0) var tex: {texture};{vertex}@fragment fn fs(@builtin(position) p:vec4<f32>{fragment_args}){{if(({sample}&{mask}u)==0u){{discard;}}}}"
        );
        let pipeline = make_pipeline(
            format!("stencil-copy-plane:{fmt}:{}:{bit}", source.desc.samples),
            code,
            mask,
        )?;
        let layout = js::call1(&pipeline, "getBindGroupLayout", &number(0u32)).map_err(error)?;
        planes.push((mask, pipeline, layout));
    }
    let alias = js_sys::Object::is(&source.raw, &target.raw);
    for layer in 0..src.extent[2] {
        let sampled = if alias {
            let td = object();
            let size = object();
            field(&size, "width", number(src.extent[0]))?;
            field(&size, "height", number(src.extent[1]))?;
            field(&size, "depthOrArrayLayers", number(1u32))?;
            field(&td, "size", size)?;
            field(&td, "sampleCount", number(source.desc.samples))?;
            field(&td, "format", JsValue::from_str(fmt))?;
            field(&td, "usage", number(2u32 | 4 | 16))?;
            let temp = js::call1(&encoder.device.0.device, "createTexture", &td).map_err(error)?;
            let from = object();
            field(&from, "texture", source.raw.clone())?;
            field(&from, "mipLevel", number(src.mip))?;
            let origin = object();
            field(&origin, "z", number(src.origin[2] + layer))?;
            field(&from, "origin", origin)?;
            let to = object();
            field(&to, "texture", temp.clone())?;
            let extent = object();
            field(&extent, "width", number(src.extent[0]))?;
            field(&extent, "height", number(src.extent[1]))?;
            field(&extent, "depthOrArrayLayers", number(1u32))?;
            js::call(&encoder.raw, "copyTextureToTexture", &[from, to, extent]).map_err(error)?;
            temp
        } else {
            source.raw.clone()
        };
        let svd = object();
        field(&svd, "dimension", JsValue::from_str("2d"))?;
        field(&svd, "aspect", JsValue::from_str("stencil-only"))?;
        field(
            &svd,
            "baseMipLevel",
            number(if alias { 0 } else { src.mip }),
        )?;
        field(&svd, "mipLevelCount", number(1u32))?;
        field(
            &svd,
            "baseArrayLayer",
            number(if alias { 0 } else { src.origin[2] + layer }),
        )?;
        field(&svd, "arrayLayerCount", number(1u32))?;
        let sv = js::call1(&sampled, "createView", &svd).map_err(error)?;
        let dvd = object();
        field(&dvd, "dimension", JsValue::from_str("2d"))?;
        field(&dvd, "baseMipLevel", number(dst.mip))?;
        field(&dvd, "mipLevelCount", number(1u32))?;
        field(&dvd, "baseArrayLayer", number(dst.origin[2] + layer))?;
        field(&dvd, "arrayLayerCount", number(1u32))?;
        let dv = js::call1(&target.raw, "createView", &dvd).map_err(error)?;
        let attachment = object();
        field(&attachment, "view", dv.clone())?;
        field(&attachment, "depthReadOnly", JsValue::TRUE)?;
        field(&attachment, "stencilLoadOp", JsValue::from_str("load"))?;
        field(&attachment, "stencilStoreOp", JsValue::from_str("store"))?;
        let pass_desc = object();
        field(&pass_desc, "colorAttachments", Array::new().into())?;
        field(&pass_desc, "depthStencilAttachment", attachment)?;
        let pass = js::call1(&encoder.raw, "beginRenderPass", &pass_desc).map_err(error)?;
        js::call1(&pass, "setPipeline", &clear).map_err(error)?;
        js::call1(&pass, "setStencilReference", &number(0u32)).map_err(error)?;
        js::call(
            &pass,
            "draw",
            &[number(3u32), number(1u32), number(0u32), number(0u32)],
        )
        .map_err(error)?;
        for (mask, pipeline, layout) in &planes {
            let entries = Array::new();
            let a = object();
            field(&a, "binding", number(0u32))?;
            field(&a, "resource", sv.clone())?;
            entries.push(&a);
            let bd = object();
            field(&bd, "layout", layout.clone())?;
            field(&bd, "entries", entries.into())?;
            let bind =
                js::call1(&encoder.device.0.device, "createBindGroup", &bd).map_err(error)?;
            js::call1(&pass, "setPipeline", pipeline).map_err(error)?;
            js::call(&pass, "setBindGroup", &[number(0u32), bind]).map_err(error)?;
            js::call1(&pass, "setStencilReference", &number(*mask)).map_err(error)?;
            js::call(
                &pass,
                "draw",
                &[number(3u32), number(1u32), number(0u32), number(0u32)],
            )
            .map_err(error)?;
        }
        js::call0(&pass, "end").map_err(error)?;
        encoder.keep_alive.extend([sampled, sv, dv]);
    }
    encoder
        .keep_alive
        .extend([source.raw.clone(), target.raw.clone(), clear]);
    for (_, pipeline, layout) in planes {
        encoder.keep_alive.extend([pipeline, layout]);
    }
    Ok(())
}
