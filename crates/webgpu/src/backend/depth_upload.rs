//! Upload float depth through a fragment shader when WebGPU forbids texel copies.
use super::{Result, WebGpu, WebGpuEncoder, api, error, field, js, number, object};
use js_sys::Array;
use wasm_bindgen::JsValue;

pub(super) fn upload(
    encoder: &mut WebGpuEncoder,
    source: &JsValue,
    offset: u64,
    row_stride: u64,
    rows_per_image: u64,
    dst: &api::TextureRegion<'_, WebGpu>,
) -> Result<()> {
    // SAFETY: the caller validated the device, single-sample format, whole
    // subresource and source bounds before invoking this helper.
    let texture = unsafe { dst.texture.native_ref() };
    let format = super::super::resource::texture_format(texture.desc.format)?;
    let width_bytes = u64::from(dst.extent[0]) * 4;
    let rows_per_chunk = (4 * 1024 * 1024 / row_stride.max(1))
        .max(1)
        .min(u64::from(dst.extent[1]));
    for layer in 0..dst.extent[2] {
        let mut row = 0u64;
        while row < u64::from(dst.extent[1]) {
            let rows = rows_per_chunk.min(u64::from(dst.extent[1]) - row);
            let source_offset = offset + (u64::from(layer) * rows_per_image + row) * row_stride;
            let repack = source_offset % 4 != 0 || row_stride % 4 != 0;
            let actual_stride = if repack { width_bytes } else { row_stride };
            let size = (rows - 1) * actual_stride + width_bytes;
            let desc = object();
            field(&desc, "size", number(size as f64))?;
            field(&desc, "usage", number(4u32 | 8 | 128))?;
            let temp = js::call1(&encoder.device.0.device, "createBuffer", &desc).map_err(error)?;
            if repack {
                let wrap = |raw: JsValue, size: u64| super::WebGpuBuffer {
                    raw,
                    size,
                    usage: api::BufferUsage::COPY_SRC.union(api::BufferUsage::COPY_DST),
                    mapped: std::rc::Rc::new(std::cell::Cell::new(false)),
                };
                let source_buffer = wrap(source.clone(), 0);
                let target_buffer = wrap(temp.clone(), size);
                for y in 0..rows {
                    super::unaligned::copy_word(
                        encoder,
                        &source_buffer,
                        source_offset + y * row_stride,
                        &target_buffer,
                        y * actual_stride,
                        width_bytes as usize,
                    )?;
                }
            } else {
                js::call(
                    &encoder.raw,
                    "copyBufferToBuffer",
                    &[
                        source.clone(),
                        number(source_offset as f64),
                        temp.clone(),
                        number(0u32),
                        number(size as f64),
                    ],
                )
                .map_err(error)?;
            }
            let code = format!(
                "@group(0) @binding(0) var<storage, read> data: array<u32>; struct O {{ @builtin(position) p: vec4<f32>, }}; @vertex fn vs(@builtin(vertex_index) i: u32) -> O {{ let uv=vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)); var o:O; o.p=vec4<f32>(uv * vec2<f32>(2., -2.) + vec2<f32>(-1., 1.), 0., 1.); return o; }} @fragment fn fs(@builtin(position) p:vec4<f32>) -> @builtin(frag_depth) f32 {{ let x=u32(p.x); let y=u32(p.y)-{row}u; return bitcast<f32>(data[y*{stride}u+x]); }}",
                stride = actual_stride / 4
            );
            let md = object();
            field(&md, "code", JsValue::from_str(&code))?;
            let module =
                js::call1(&encoder.device.0.device, "createShaderModule", &md).map_err(error)?;
            let vertex = object();
            field(&vertex, "module", module.clone())?;
            field(&vertex, "entryPoint", JsValue::from_str("vs"))?;
            let fragment = object();
            field(&fragment, "module", module)?;
            field(&fragment, "entryPoint", JsValue::from_str("fs"))?;
            field(&fragment, "targets", Array::new().into())?;
            let ds = object();
            field(&ds, "format", JsValue::from_str(format))?;
            field(&ds, "depthWriteEnabled", JsValue::TRUE)?;
            field(&ds, "depthCompare", JsValue::from_str("always"))?;
            let pd = object();
            field(&pd, "layout", JsValue::from_str("auto"))?;
            field(&pd, "vertex", vertex)?;
            field(&pd, "fragment", fragment)?;
            field(&pd, "depthStencil", ds)?;
            let pipeline =
                js::call1(&encoder.device.0.device, "createRenderPipeline", &pd).map_err(error)?;
            let layout =
                js::call1(&pipeline, "getBindGroupLayout", &number(0u32)).map_err(error)?;
            let r = object();
            field(&r, "buffer", temp.clone())?;
            let e = object();
            field(&e, "binding", number(0u32))?;
            field(&e, "resource", r)?;
            let entries = Array::new();
            entries.push(&e);
            let bd = object();
            field(&bd, "layout", layout)?;
            field(&bd, "entries", entries.into())?;
            let bind =
                js::call1(&encoder.device.0.device, "createBindGroup", &bd).map_err(error)?;
            let vd = object();
            field(&vd, "dimension", JsValue::from_str("2d"))?;
            field(&vd, "baseMipLevel", number(dst.mip))?;
            field(&vd, "mipLevelCount", number(1u32))?;
            field(&vd, "baseArrayLayer", number(dst.origin[2] + layer))?;
            field(&vd, "arrayLayerCount", number(1u32))?;
            let view = js::call1(&texture.raw, "createView", &vd).map_err(error)?;
            let a = object();
            field(&a, "view", view.clone())?;
            field(&a, "depthLoadOp", JsValue::from_str("load"))?;
            field(&a, "depthStoreOp", JsValue::from_str("store"))?;
            if format.contains("stencil") {
                field(&a, "stencilReadOnly", JsValue::TRUE)?;
            }
            let pass_desc = object();
            field(&pass_desc, "colorAttachments", Array::new().into())?;
            field(&pass_desc, "depthStencilAttachment", a)?;
            let pass = js::call1(&encoder.raw, "beginRenderPass", &pass_desc).map_err(error)?;
            js::call1(&pass, "setPipeline", &pipeline).map_err(error)?;
            js::call(&pass, "setBindGroup", &[number(0u32), bind.clone()]).map_err(error)?;
            js::call(
                &pass,
                "setScissorRect",
                &[
                    number(0u32),
                    number(row as f64),
                    number(dst.extent[0]),
                    number(rows as f64),
                ],
            )
            .map_err(error)?;
            js::call(
                &pass,
                "draw",
                &[number(3u32), number(1u32), number(0u32), number(0u32)],
            )
            .map_err(error)?;
            js::call0(&pass, "end").map_err(error)?;
            encoder.keep_alive.extend([temp, pipeline, bind, view]);
            row += rows;
        }
    }
    encoder
        .keep_alive
        .extend([source.clone(), texture.raw.clone()]);
    Ok(())
}
