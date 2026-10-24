//! Native scissored clears for render areas smaller than their attachments.

use super::{
    command::{WebGpuEncoder, error, field, number, object},
    instance::WebGpu,
    js, resource,
};
use fluxel_rhi_core::api::{self, Result};
use js_sys::Array;
use js_sys::Uint8Array;
use wasm_bindgen::JsValue;

pub(super) fn view_info(
    view: &api::AttachmentView<'_, WebGpu>,
) -> (JsValue, api::TextureFormat, [u32; 2], u32) {
    match view {
        api::AttachmentView::Texture(view) => {
            // SAFETY: Core validates attachment ownership and lifetime before backend entry.
            let native = unsafe { view.native_ref() };
            let tex = &native._texture.desc;
            let mip = native.desc.mips.start;
            (
                native.raw.clone(),
                native.desc.format,
                [(tex.extent[0] >> mip).max(1), (tex.extent[1] >> mip).max(1)],
                tex.samples,
            )
        }
        api::AttachmentView::Surface(image) => {
            // SAFETY: Core validates the current image lease before backend entry.
            let native = unsafe { image.native_ref() };
            (native.view.clone(), native.format, native.extent, 1)
        }
    }
}

fn create_pipeline(
    encoder: &WebGpuEncoder,
    format: api::TextureFormat,
    samples: u32,
    code: &str,
    depth: bool,
    stencil: bool,
) -> Result<JsValue> {
    let shader = object();
    field(&shader, "code", JsValue::from_str(code))?;
    let module =
        js::call1(&encoder.device.0.device, "createShaderModule", &shader).map_err(error)?;
    let vertex = object();
    field(&vertex, "module", module.clone())?;
    field(&vertex, "entryPoint", JsValue::from_str("vs"))?;
    let frag = object();
    field(&frag, "module", module)?;
    field(&frag, "entryPoint", JsValue::from_str("fs"))?;
    let targets = Array::new();
    if !depth && !stencil {
        let target = object();
        field(
            &target,
            "format",
            JsValue::from_str(resource::texture_format(format)?),
        )?;
        targets.push(&target);
    }
    field(&frag, "targets", targets.into())?;
    let desc = object();
    field(&desc, "layout", JsValue::from_str("auto"))?;
    field(&desc, "vertex", vertex)?;
    field(&desc, "fragment", frag)?;
    let multisample = object();
    field(&multisample, "count", number(samples))?;
    field(&desc, "multisample", multisample)?;
    if depth || stencil {
        let state = object();
        field(
            &state,
            "format",
            JsValue::from_str(resource::texture_format(format)?),
        )?;
        field(&state, "depthWriteEnabled", JsValue::from_bool(depth))?;
        field(&state, "depthCompare", JsValue::from_str("always"))?;
        if stencil {
            let face = object();
            field(&face, "compare", JsValue::from_str("always"))?;
            field(&face, "passOp", JsValue::from_str("replace"))?;
            field(&state, "stencilFront", face.clone())?;
            field(&state, "stencilBack", face)?;
        } else {
            field(&state, "stencilWriteMask", number(0u32))?;
        }
        field(&desc, "depthStencil", state)?;
    }
    js::call1(&encoder.device.0.device, "createRenderPipeline", &desc).map_err(error)
}

fn pipeline(
    encoder: &WebGpuEncoder,
    key: String,
    format: api::TextureFormat,
    samples: u32,
    code: String,
    depth: bool,
    stencil: bool,
) -> Result<JsValue> {
    super::utility::render_pipeline(&encoder.device, key, || {
        create_pipeline(encoder, format, samples, &code, depth, stencil)
    })
}

fn uniform(encoder: &mut WebGpuEncoder, bytes: &[u8]) -> Result<JsValue> {
    let desc = object();
    field(&desc, "size", number(bytes.len() as u32))?;
    // GPUBufferUsage::UNIFORM; mapping avoids a queue-side operation between
    // command encoding steps and makes this value an encoder-local resource.
    field(&desc, "usage", number(64u32))?;
    field(&desc, "mappedAtCreation", JsValue::TRUE)?;
    let buffer = js::call1(&encoder.device.0.device, "createBuffer", &desc).map_err(error)?;
    let mapped = js::call0(&buffer, "getMappedRange").map_err(error)?;
    Uint8Array::new(&mapped).copy_from(bytes);
    js::call0(&buffer, "unmap").map_err(error)?;
    encoder.keep_alive.push(buffer.clone());
    Ok(buffer)
}

fn uniform_group(encoder: &mut WebGpuEncoder, pipe: &JsValue, bytes: &[u8]) -> Result<JsValue> {
    let buffer = uniform(encoder, bytes)?;
    let resource = object();
    field(&resource, "buffer", buffer)?;
    let entry = object();
    field(&entry, "binding", number(0u32))?;
    field(&entry, "resource", resource)?;
    let entries = Array::new();
    entries.push(&entry);
    let desc = object();
    field(
        &desc,
        "layout",
        js::call1(pipe, "getBindGroupLayout", &number(0u32)).map_err(error)?,
    )?;
    field(&desc, "entries", entries.into())?;
    js::call1(&encoder.device.0.device, "createBindGroup", &desc).map_err(error)
}

const VERTEX: &str = "@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(3.0,-1.0),vec2f(-1.0,3.0)); return vec4f(p[i],0.0,1.0); }\n";

fn draw(
    encoder: &mut WebGpuEncoder,
    attachment: JsValue,
    depth_attachment: bool,
    pipe: JsValue,
    group: Option<JsValue>,
    area: api::Rect,
    stencil: Option<u32>,
) -> Result<()> {
    let pass_desc = object();
    let colors = Array::new();
    if depth_attachment {
        field(&pass_desc, "depthStencilAttachment", attachment)?;
    } else {
        colors.push(&attachment);
    }
    field(&pass_desc, "colorAttachments", colors.into())?;
    let pass = js::call1(&encoder.raw, "beginRenderPass", &pass_desc).map_err(error)?;
    js::call1(&pass, "setPipeline", &pipe).map_err(error)?;
    if let Some(group) = &group {
        js::call(&pass, "setBindGroup", &[number(0u32), group.clone()]).map_err(error)?;
    }
    js::call(
        &pass,
        "setScissorRect",
        &[
            number(area.x),
            number(area.y),
            number(area.width),
            number(area.height),
        ],
    )
    .map_err(error)?;
    if let Some(value) = stencil {
        js::call1(&pass, "setStencilReference", &number(value)).map_err(error)?;
    }
    js::call(&pass, "draw", &[number(3u32)]).map_err(error)?;
    js::call0(&pass, "end").map_err(error)?;
    encoder.keep_alive.push(pipe);
    if let Some(group) = group {
        encoder.keep_alive.push(group);
    }
    Ok(())
}

pub(super) fn clear(
    encoder: &mut WebGpuEncoder,
    desc: &api::RasterPassDesc<'_, WebGpu>,
) -> Result<()> {
    let area = desc.render_area;
    for color in desc.colors.iter().flatten() {
        let api::LoadOp::Clear(value) = color.load else {
            continue;
        };
        let (view, format, _, samples) = view_info(&color.view);
        let (ty, kind, bytes) = match value {
            api::ClearColor::Float(v) => (
                "vec4f",
                "float",
                v.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>(),
            ),
            api::ClearColor::Uint(v) => (
                "vec4u",
                "uint",
                v.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>(),
            ),
            api::ClearColor::Sint(v) => (
                "vec4i",
                "sint",
                v.into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>(),
            ),
        };
        let code = format!(
            "{VERTEX}@group(0) @binding(0) var<uniform> value: {ty}; @fragment fn fs()->@location(0) {ty} {{ return value; }}"
        );
        let pipe = pipeline(
            encoder,
            format!("partial-clear-color:{kind}:{format:?}:{samples}"),
            format,
            samples,
            code,
            false,
            false,
        )?;
        let group = uniform_group(encoder, &pipe, &bytes)?;
        let attachment = object();
        field(&attachment, "view", view.clone())?;
        field(&attachment, "loadOp", JsValue::from_str("load"))?;
        field(&attachment, "storeOp", JsValue::from_str("store"))?;
        draw(encoder, attachment, false, pipe, Some(group), area, None)?;
        encoder.keep_alive.push(view);
    }
    if let Some(depth) = &desc.depth_stencil {
        let depth_clear = depth.depth.and_then(|(load, _)| {
            if let api::LoadOp::Clear(value) = load {
                Some(value)
            } else {
                None
            }
        });
        let stencil_clear = depth.stencil.and_then(|(load, _)| {
            if let api::LoadOp::Clear(value) = load {
                Some(value)
            } else {
                None
            }
        });
        if depth_clear.is_some() || stencil_clear.is_some() {
            let (view, format, _, samples) = view_info(&depth.view);
            let fragment = if depth_clear.is_some() {
                "@group(0) @binding(0) var<uniform> value: f32; @fragment fn fs()->@builtin(frag_depth) f32 { return value; }".to_string()
            } else {
                "@fragment fn fs() {}".to_string()
            };
            let pipe = pipeline(
                encoder,
                format!(
                    "partial-clear-depth:{format:?}:{samples}:{}:{}",
                    depth_clear.is_some(),
                    stencil_clear.is_some()
                ),
                format,
                samples,
                format!("{VERTEX}{fragment}"),
                depth_clear.is_some(),
                stencil_clear.is_some(),
            )?;
            let group = depth_clear
                .map(|value| uniform_group(encoder, &pipe, &value.to_le_bytes()))
                .transpose()?;
            let attachment = object();
            field(&attachment, "view", view.clone())?;
            let has_depth = !matches!(format, api::TextureFormat::Stencil8);
            let has_stencil = matches!(
                format,
                api::TextureFormat::Stencil8
                    | api::TextureFormat::Depth24PlusStencil8
                    | api::TextureFormat::Depth32FloatStencil8
            );
            if has_depth {
                if depth_clear.is_some() {
                    field(&attachment, "depthLoadOp", JsValue::from_str("load"))?;
                    field(&attachment, "depthStoreOp", JsValue::from_str("store"))?;
                } else {
                    field(&attachment, "depthReadOnly", JsValue::TRUE)?;
                }
            }
            if has_stencil {
                if stencil_clear.is_some() {
                    field(&attachment, "stencilLoadOp", JsValue::from_str("load"))?;
                    field(&attachment, "stencilStoreOp", JsValue::from_str("store"))?;
                } else {
                    field(&attachment, "stencilReadOnly", JsValue::TRUE)?;
                }
            }
            draw(encoder, attachment, true, pipe, group, area, stencil_clear)?;
            encoder.keep_alive.push(view);
        }
    }
    Ok(())
}

/// A native resolve target and the subsequent scissored transfer into the actual attachment.
pub(super) struct PartialResolve {
    source_view: JsValue,
    destination_view: JsValue,
    format: api::TextureFormat,
    area: api::Rect,
}

pub(super) fn prepare_resolve(
    encoder: &mut WebGpuEncoder,
    target: &api::AttachmentView<'_, WebGpu>,
    area: api::Rect,
) -> Result<(JsValue, PartialResolve)> {
    let (destination_view, format, extent, _) = view_info(target);
    let desc = object();
    let size = object();
    field(&size, "width", number(extent[0]))?;
    field(&size, "height", number(extent[1]))?;
    field(&size, "depthOrArrayLayers", number(1u32))?;
    field(&desc, "size", size)?;
    field(
        &desc,
        "format",
        JsValue::from_str(resource::texture_format(format)?),
    )?;
    // RENDER_ATTACHMENT | TEXTURE_BINDING; the canvas need only be renderable.
    field(&desc, "usage", number(20u32))?;
    let texture = js::call1(&encoder.device.0.device, "createTexture", &desc).map_err(error)?;
    let source_view = js::call1(&texture, "createView", &object()).map_err(error)?;
    encoder
        .keep_alive
        .extend([texture, source_view.clone(), destination_view.clone()]);
    Ok((
        source_view.clone(),
        PartialResolve {
            source_view,
            destination_view,
            format,
            area,
        },
    ))
}

pub(super) fn finish_resolve(encoder: &mut WebGpuEncoder, pending: PartialResolve) -> Result<()> {
    let origin = [pending.area.x, pending.area.y];
    finish_resolve_region(
        encoder,
        pending.source_view,
        pending.destination_view,
        pending.format,
        pending.area,
        origin,
    )
}

/// Copies an already resolved scratch image into an exact attachment rectangle.
/// The native draw is recorded immediately; outside pixels retain their contents.
pub(super) fn finish_resolve_region(
    encoder: &mut WebGpuEncoder,
    source_view: JsValue,
    destination_view: JsValue,
    format: api::TextureFormat,
    area: api::Rect,
    source_origin: [u32; 2],
) -> Result<()> {
    let pending = PartialResolve {
        source_view,
        destination_view,
        format,
        area,
    };
    let dx = i64::from(source_origin[0]) - i64::from(area.x);
    let dy = i64::from(source_origin[1]) - i64::from(area.y);
    let code = format!(
        "{VERTEX}@group(0) @binding(0) var source: texture_2d<f32>; @group(0) @binding(1) var<uniform> source_offset: vec2i; @fragment fn fs(@builtin(position) p: vec4f)->@location(0) vec4f {{ return textureLoad(source, vec2i(p.xy) + source_offset, 0); }}"
    );
    let pipe = pipeline(
        encoder,
        format!("partial-resolve-color:{:?}", pending.format),
        pending.format,
        1,
        code,
        false,
        false,
    )?;
    let layout = js::call1(&pipe, "getBindGroupLayout", &number(0u32)).map_err(error)?;
    let entry = object();
    field(&entry, "binding", number(0u32))?;
    field(&entry, "resource", pending.source_view.clone())?;
    let entries = Array::new();
    entries.push(&entry);
    let mut offset = Vec::with_capacity(8);
    offset.extend_from_slice(&(dx as i32).to_le_bytes());
    offset.extend_from_slice(&(dy as i32).to_le_bytes());
    let buffer = uniform(encoder, &offset)?;
    let uniform_resource = object();
    field(&uniform_resource, "buffer", buffer)?;
    let uniform_entry = object();
    field(&uniform_entry, "binding", number(1u32))?;
    field(&uniform_entry, "resource", uniform_resource)?;
    entries.push(&uniform_entry);
    let group_desc = object();
    field(&group_desc, "layout", layout)?;
    field(&group_desc, "entries", entries.into())?;
    let group =
        js::call1(&encoder.device.0.device, "createBindGroup", &group_desc).map_err(error)?;
    let attachment = object();
    field(&attachment, "view", pending.destination_view.clone())?;
    field(&attachment, "loadOp", JsValue::from_str("load"))?;
    field(&attachment, "storeOp", JsValue::from_str("store"))?;
    let colors = Array::new();
    colors.push(&attachment);
    let pass_desc = object();
    field(&pass_desc, "colorAttachments", colors.into())?;
    let pass = js::call1(&encoder.raw, "beginRenderPass", &pass_desc).map_err(error)?;
    js::call1(&pass, "setPipeline", &pipe).map_err(error)?;
    js::call(&pass, "setBindGroup", &[number(0u32), group.clone()]).map_err(error)?;
    js::call(
        &pass,
        "setScissorRect",
        &[
            number(pending.area.x),
            number(pending.area.y),
            number(pending.area.width),
            number(pending.area.height),
        ],
    )
    .map_err(error)?;
    js::call(&pass, "draw", &[number(3u32)]).map_err(error)?;
    js::call0(&pass, "end").map_err(error)?;
    encoder.keep_alive.extend([pipe, group]);
    Ok(())
}
