//! Direct WebGPU raster-pass and draw lowering.
//!
//! Core admits attachment and query-set DeviceIds before `begin_raster`. This
//! module turns that single live pass into WebGPU calls and retains every raw
//! attachment identity in the owning command encoder through queue completion.

use std::ops::Range;

use fluxel_rhi_core::api::{self, Error, Result};
use js_sys::Array;
use wasm_bindgen::JsValue;

use super::{
    command::{WebGpuEncoder, error, field, number, object, pass},
    instance::WebGpu,
    js,
    resource::WebGpuBuffer,
};

fn transient_intent(view: &api::AttachmentView<'_, WebGpu>) -> bool {
    match view {
        api::AttachmentView::Texture(view) => {
            // SAFETY: Core admitted this view's device before raster lowering.
            unsafe { view.native_ref() }
                ._texture
                .desc
                .usage
                .contains(api::TextureUsage::TRANSIENT_ATTACHMENT)
        }
        api::AttachmentView::Surface(_) => false,
    }
}

impl fluxel_rhi_core::backend::RasterBackend for WebGpu {
    fn begin_raster(
        encoder: &mut WebGpuEncoder,
        desc: &api::RasterPassDesc<'_, Self>,
    ) -> Result<()> {
        if encoder.pass.is_some()
            || desc.multiview_mask.is_some()
            || desc.secondary_inheritance.is_some()
        {
            return Err(Error::Unsupported("WebGPU raster pass option"));
        }
        if let Some(depth) = &desc.depth_stencil {
            if depth.depth_read_only
                && depth.depth.is_some_and(|(load, store)| {
                    !matches!(load, api::LoadOp::Load) || !matches!(store, api::StoreOp::Store)
                })
            {
                return Err(Error::InvalidState);
            }
            if depth.stencil_read_only
                && depth.stencil.is_some_and(|(load, store)| {
                    !matches!(load, api::LoadOp::Load) || !matches!(store, api::StoreOp::Store)
                })
            {
                return Err(Error::InvalidState);
            }
        }
        for view in desc
            .colors
            .iter()
            .flatten()
            .flat_map(|color| std::iter::once(&color.view).chain(color.resolve_target.iter()))
            .chain(desc.depth_stencil.iter().map(|depth| &depth.view))
        {
            if let api::AttachmentView::Surface(image) = view {
                // SAFETY: Core validated the image lease identity before entering the backend.
                let usage = unsafe { image.native_ref() }.usage;
                if usage.0 & api::SurfaceUsage::RENDER_ATTACHMENT.0 == 0 {
                    return Err(Error::InvalidRange);
                }
            }
        }
        let area = desc.render_area;
        let right = area.x.checked_add(area.width).ok_or(Error::InvalidRange)?;
        let bottom = area.y.checked_add(area.height).ok_or(Error::InvalidRange)?;
        let mut partial = false;
        for view in desc
            .colors
            .iter()
            .flatten()
            .map(|color| &color.view)
            .chain(desc.depth_stencil.iter().map(|depth| &depth.view))
        {
            let (_, _, extent, _) = super::partial::view_info(view);
            if right > extent[0] || bottom > extent[1] {
                return Err(Error::InvalidRange);
            }
            partial |= area.x != 0 || area.y != 0 || right != extent[0] || bottom != extent[1];
        }
        let transient = desc.colors.iter().flatten().any(|color| {
            transient_intent(&color.view)
                || color.resolve_target.as_ref().is_some_and(transient_intent)
        }) || desc
            .depth_stencil
            .as_ref()
            .is_some_and(|depth| transient_intent(&depth.view));
        if partial && transient {
            return Err(Error::Unsupported("partial transient attachment pass"));
        }
        if partial {
            if let Some(writes) = &desc.timestamp_writes {
                // SAFETY: Core validated the query set against this encoder.
                let set = unsafe { writes.set.native_ref() }.clone();
                if let Some(begin) = writes.begin {
                    <WebGpu as fluxel_rhi_core::backend::QueryBackend>::write_timestamp(
                        encoder, &set, begin,
                    )?;
                }
                encoder.partial_timestamp_end = writes.end.map(|end| (set, end));
            }
            super::partial::clear(encoder, desc)?;
        }
        let native = object();
        if let Some(label) = desc.label {
            field(&native, "label", JsValue::from_str(label))?;
        }
        let colors = Array::new();
        for color in desc.colors {
            let Some(color) = color else {
                colors.push(&JsValue::NULL);
                continue;
            };
            let attachment = object();
            let view = match &color.view {
                api::AttachmentView::Texture(view) => {
                    // SAFETY: `CommandEncoder::begin_raster` checked this view's
                    // DeviceId; its cloned JS view enters `keep_alive` below.
                    unsafe { view.native_ref() }.raw.clone()
                }
                api::AttachmentView::Surface(image) => {
                    // SAFETY: `begin_raster` checked this lease's DeviceId and
                    // current generation; only its cloned view is retained.
                    unsafe { image.native_ref() }.view.clone()
                }
            };
            field(&attachment, "view", view.clone())?;
            encoder.keep_alive.push(view);
            if let Some(resolve) = &color.resolve_target {
                let target = match resolve {
                    api::AttachmentView::Texture(view) => {
                        // SAFETY: `begin_raster` checked this view's DeviceId;
                        // its cloned JS view enters `keep_alive` below.
                        unsafe { view.native_ref() }.raw.clone()
                    }
                    api::AttachmentView::Surface(image) => {
                        // SAFETY: `begin_raster` checked this lease's DeviceId
                        // and current generation before entering this SPI call.
                        unsafe { image.native_ref() }.view.clone()
                    }
                };
                let target = if partial {
                    let (scratch, pending) =
                        super::partial::prepare_resolve(encoder, resolve, area)?;
                    encoder.partial_resolves.push(pending);
                    scratch
                } else {
                    target
                };
                field(&attachment, "resolveTarget", target.clone())?;
                encoder.keep_alive.push(target);
            }
            let load = if partial {
                "load"
            } else {
                match color.load {
                    api::LoadOp::Load => "load",
                    api::LoadOp::Clear(_) | api::LoadOp::Discard => "clear",
                }
            };
            field(&attachment, "loadOp", JsValue::from_str(load))?;
            if let api::LoadOp::Clear(clear) = color.load {
                let rgba = match clear {
                    api::ClearColor::Float(value) => value.map(f64::from),
                    api::ClearColor::Uint(value) => value.map(f64::from),
                    api::ClearColor::Sint(value) => value.map(f64::from),
                };
                let value = object();
                for (channel, component) in ["r", "g", "b", "a"].into_iter().zip(rgba) {
                    field(&value, channel, number(component))?;
                }
                field(&attachment, "clearValue", value)?;
            } else if matches!(color.load, api::LoadOp::Discard) {
                let value = object();
                for channel in ["r", "g", "b", "a"] {
                    field(&value, channel, number(0_u32))?;
                }
                field(&attachment, "clearValue", value)?;
            }
            field(
                &attachment,
                "storeOp",
                JsValue::from_str(match color.store {
                    api::StoreOp::Store => "store",
                    api::StoreOp::Discard => "discard",
                }),
            )?;
            colors.push(&attachment);
        }
        field(&native, "colorAttachments", colors.into())?;
        if desc.depth_stencil.is_none() && desc.colors.iter().all(Option::is_none) {
            let texture_desc = object();
            let size = object();
            field(&size, "width", number(right.max(1)))?;
            field(&size, "height", number(bottom.max(1)))?;
            field(&size, "depthOrArrayLayers", number(1_u32))?;
            field(&texture_desc, "size", size)?;
            field(&texture_desc, "format", JsValue::from_str("depth32float"))?;
            field(&texture_desc, "usage", number(16_u32))?;
            let texture = js::call1(&encoder.device.0.device, "createTexture", &texture_desc)
                .map_err(error)?;
            let view = js::call1(&texture, "createView", &object()).map_err(error)?;
            let attachment = object();
            field(&attachment, "view", view.clone())?;
            field(&attachment, "depthLoadOp", JsValue::from_str("clear"))?;
            field(&attachment, "depthClearValue", number(1_f64))?;
            field(&attachment, "depthStoreOp", JsValue::from_str("discard"))?;
            field(&native, "depthStencilAttachment", attachment)?;
            encoder.keep_alive.extend([texture, view]);
        }
        if let Some(set) = desc.occlusion_query_set {
            // SAFETY: `begin_raster` checked the query set DeviceId; this clone
            // is retained by the native encoder until queue completion.
            let raw = unsafe { set.native_ref() }.raw.clone();
            field(&native, "occlusionQuerySet", raw.clone())?;
            encoder.keep_alive.push(raw);
        }
        if let Some(depth) = &desc.depth_stencil {
            let attachment = object();
            let view = match &depth.view {
                api::AttachmentView::Texture(view) => {
                    // SAFETY: `begin_raster` checked this view's DeviceId; the
                    // JS clone is retained by the native encoder below.
                    unsafe { view.native_ref() }.raw.clone()
                }
                api::AttachmentView::Surface(image) => {
                    // SAFETY: `begin_raster` checked this lease's DeviceId and
                    // current generation before the pass descriptor was built.
                    unsafe { image.native_ref() }.view.clone()
                }
            };
            field(&attachment, "view", view.clone())?;
            encoder.keep_alive.push(view);
            if let Some((load, store)) = depth.depth {
                if depth.depth_read_only {
                    if !matches!(load, api::LoadOp::Load) || !matches!(store, api::StoreOp::Store) {
                        return Err(Error::InvalidState);
                    }
                    field(&attachment, "depthReadOnly", JsValue::TRUE)?;
                } else {
                    field(
                        &attachment,
                        "depthLoadOp",
                        JsValue::from_str(if partial {
                            "load"
                        } else {
                            match load {
                                api::LoadOp::Load => "load",
                                api::LoadOp::Clear(_) | api::LoadOp::Discard => "clear",
                            }
                        }),
                    )?;
                    match load {
                        api::LoadOp::Clear(value) => {
                            field(&attachment, "depthClearValue", number(value as f64))?
                        }
                        api::LoadOp::Discard => {
                            field(&attachment, "depthClearValue", number(1.0_f64))?
                        }
                        api::LoadOp::Load => {}
                    }
                    field(
                        &attachment,
                        "depthStoreOp",
                        JsValue::from_str(match store {
                            api::StoreOp::Store => "store",
                            api::StoreOp::Discard => "discard",
                        }),
                    )?;
                }
            }
            if depth.depth.is_none() {
                field(&attachment, "depthReadOnly", JsValue::TRUE)?;
            }
            if let Some((load, store)) = depth.stencil {
                if depth.stencil_read_only {
                    if !matches!(load, api::LoadOp::Load) || !matches!(store, api::StoreOp::Store) {
                        return Err(Error::InvalidState);
                    }
                    field(&attachment, "stencilReadOnly", JsValue::TRUE)?;
                } else {
                    field(
                        &attachment,
                        "stencilLoadOp",
                        JsValue::from_str(if partial {
                            "load"
                        } else {
                            match load {
                                api::LoadOp::Load => "load",
                                api::LoadOp::Clear(_) | api::LoadOp::Discard => "clear",
                            }
                        }),
                    )?;
                    match load {
                        api::LoadOp::Clear(value) => {
                            field(&attachment, "stencilClearValue", number(value))?
                        }
                        api::LoadOp::Discard => {
                            field(&attachment, "stencilClearValue", number(0_u32))?
                        }
                        api::LoadOp::Load => {}
                    }
                    field(
                        &attachment,
                        "stencilStoreOp",
                        JsValue::from_str(match store {
                            api::StoreOp::Store => "store",
                            api::StoreOp::Discard => "discard",
                        }),
                    )?;
                }
            }
            if depth.stencil.is_none() {
                field(&attachment, "stencilReadOnly", JsValue::TRUE)?;
            }
            field(&native, "depthStencilAttachment", attachment)?;
        }
        if let Some(writes) = desc.timestamp_writes.as_ref().filter(|_| !partial) {
            let timestamp = object();
            // SAFETY: Core checked the query set's device identity.
            let raw = unsafe { writes.set.native_ref() }.raw.clone();
            field(&timestamp, "querySet", raw.clone())?;
            if let Some(index) = writes.begin {
                field(&timestamp, "beginningOfPassWriteIndex", number(index))?;
            }
            if let Some(index) = writes.end {
                field(&timestamp, "endOfPassWriteIndex", number(index))?;
            }
            field(&native, "timestampWrites", timestamp)?;
            encoder.keep_alive.push(raw);
        }
        encoder.pass = Some(js::call1(&encoder.raw, "beginRenderPass", &native).map_err(error)?);
        let dummy = desc.depth_stencil.is_none() && desc.colors.iter().all(Option::is_none);
        super::pass_split::begin(pass(encoder)?, &native, right, bottom, dummy)?;
        encoder.render_area = Some(area);
        Self::scissor(encoder, area)?;
        encoder.occlusion_query_set = desc.occlusion_query_set.map(|set| {
            // SAFETY: `begin_raster` checked the query set DeviceId; this clone
            // only identifies the current pass until `end_raster` clears it.
            unsafe { set.native_ref() }.raw.clone()
        });
        encoder.occlusion_logical_set = encoder.occlusion_query_set.clone();
        encoder.occlusion_next_index = 0;
        Ok(())
    }

    fn set_pipeline(encoder: &mut WebGpuEncoder, pipeline: &JsValue) -> Result<()> {
        let samples = super::immediate::samples(pipeline)?;
        super::pass_split::split(encoder, samples)?;
        js::call1(pass(encoder)?, "setPipeline", pipeline).map_err(error)?;
        super::immediate::bind_pipeline(pass(encoder)?, pipeline, false)?;
        super::pass_split::record(
            pass(encoder)?,
            "pipeline",
            "setPipeline",
            &[pipeline.clone()],
        )?;
        encoder.keep_alive.push(pipeline.clone());
        Ok(())
    }

    fn set_bind_group(
        encoder: &mut WebGpuEncoder,
        slot: u32,
        group: &JsValue,
        offsets: &[u32],
    ) -> Result<()> {
        let dynamic = Array::new();
        for offset in offsets {
            dynamic.push(&number(*offset));
        }
        js::call(
            pass(encoder)?,
            "setBindGroup",
            &[number(slot), group.clone(), dynamic.clone().into()],
        )
        .map_err(error)?;
        encoder.keep_alive.push(group.clone());
        super::pass_split::record(
            pass(encoder)?,
            &format!("group:{slot}"),
            "setBindGroup",
            &[number(slot), group.clone(), dynamic.into()],
        )?;
        Ok(())
    }

    fn set_vertex_buffer(
        encoder: &mut WebGpuEncoder,
        slot: u32,
        buffer: &WebGpuBuffer,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        if offset.checked_add(size).is_none_or(|end| end > buffer.size) {
            return Err(Error::InvalidRange);
        }
        js::call(
            pass(encoder)?,
            "setVertexBuffer",
            &[
                number(slot),
                buffer.raw.clone(),
                number(offset as f64),
                number(size as f64),
            ],
        )
        .map_err(error)?;
        encoder.keep_alive.push(buffer.raw.clone());
        super::pass_split::record(
            pass(encoder)?,
            &format!("vertex:{slot}"),
            "setVertexBuffer",
            &[
                number(slot),
                buffer.raw.clone(),
                number(offset as f64),
                number(size as f64),
            ],
        )?;
        Ok(())
    }

    fn set_index_buffer(
        encoder: &mut WebGpuEncoder,
        buffer: &WebGpuBuffer,
        offset: u64,
        size: u64,
        format: api::IndexFormat,
    ) -> Result<()> {
        if offset.checked_add(size).is_none_or(|end| end > buffer.size) {
            return Err(Error::InvalidRange);
        }
        js::call(
            pass(encoder)?,
            "setIndexBuffer",
            &[
                buffer.raw.clone(),
                JsValue::from_str(match format {
                    api::IndexFormat::Uint16 => "uint16",
                    api::IndexFormat::Uint32 => "uint32",
                }),
                number(offset as f64),
                number(size as f64),
            ],
        )
        .map_err(error)?;
        encoder.keep_alive.push(buffer.raw.clone());
        super::pass_split::record(
            pass(encoder)?,
            "index",
            "setIndexBuffer",
            &[
                buffer.raw.clone(),
                JsValue::from_str(match format {
                    api::IndexFormat::Uint16 => "uint16",
                    api::IndexFormat::Uint32 => "uint32",
                }),
                number(offset as f64),
                number(size as f64),
            ],
        )?;
        Ok(())
    }

    fn viewport(encoder: &mut WebGpuEncoder, viewport: api::Viewport) -> Result<()> {
        js::call(
            pass(encoder)?,
            "setViewport",
            &[
                number(viewport.x as f64),
                number(viewport.y as f64),
                number(viewport.width as f64),
                number(viewport.height as f64),
                number(viewport.min_depth as f64),
                number(viewport.max_depth as f64),
            ],
        )
        .map_err(error)?;
        super::pass_split::record(
            pass(encoder)?,
            "viewport",
            "setViewport",
            &[
                number(viewport.x as f64),
                number(viewport.y as f64),
                number(viewport.width as f64),
                number(viewport.height as f64),
                number(viewport.min_depth as f64),
                number(viewport.max_depth as f64),
            ],
        )?;
        Ok(())
    }

    fn scissor(encoder: &mut WebGpuEncoder, rect: api::Rect) -> Result<()> {
        let area = encoder.render_area.ok_or(Error::InvalidState)?;
        let x = rect.x.max(area.x).min(area.x.saturating_add(area.width));
        let y = rect.y.max(area.y).min(area.y.saturating_add(area.height));
        let right = rect
            .x
            .saturating_add(rect.width)
            .min(area.x.saturating_add(area.width))
            .max(x);
        let bottom = rect
            .y
            .saturating_add(rect.height)
            .min(area.y.saturating_add(area.height))
            .max(y);
        let rect = api::Rect {
            x,
            y,
            width: right - x,
            height: bottom - y,
        };
        js::call(
            pass(encoder)?,
            "setScissorRect",
            &[
                number(rect.x),
                number(rect.y),
                number(rect.width),
                number(rect.height),
            ],
        )
        .map_err(error)?;
        super::pass_split::record(
            pass(encoder)?,
            "scissor",
            "setScissorRect",
            &[
                number(rect.x),
                number(rect.y),
                number(rect.width),
                number(rect.height),
            ],
        )?;
        Ok(())
    }

    fn blend_constant(encoder: &mut WebGpuEncoder, value: [f32; 4]) -> Result<()> {
        let color = object();
        for (name, component) in ["r", "g", "b", "a"].into_iter().zip(value) {
            field(&color, name, number(component as f64))?;
        }
        js::call1(pass(encoder)?, "setBlendConstant", &color).map_err(error)?;
        super::pass_split::record(pass(encoder)?, "blend", "setBlendConstant", &[color])?;
        Ok(())
    }

    fn stencil_reference(encoder: &mut WebGpuEncoder, value: u32) -> Result<()> {
        js::call1(pass(encoder)?, "setStencilReference", &number(value)).map_err(error)?;
        super::pass_split::record(
            pass(encoder)?,
            "stencil",
            "setStencilReference",
            &[number(value)],
        )?;
        Ok(())
    }

    fn draw(
        encoder: &mut WebGpuEncoder,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) -> Result<()> {
        js::call(
            pass(encoder)?,
            "draw",
            &[
                number(
                    vertices
                        .end
                        .checked_sub(vertices.start)
                        .ok_or(Error::InvalidRange)?,
                ),
                number(
                    instances
                        .end
                        .checked_sub(instances.start)
                        .ok_or(Error::InvalidRange)?,
                ),
                number(vertices.start),
                number(instances.start),
            ],
        )
        .map_err(error)?;
        Ok(())
    }

    fn draw_indexed(
        encoder: &mut WebGpuEncoder,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> Result<()> {
        js::call(
            pass(encoder)?,
            "drawIndexed",
            &[
                number(
                    indices
                        .end
                        .checked_sub(indices.start)
                        .ok_or(Error::InvalidRange)?,
                ),
                number(
                    instances
                        .end
                        .checked_sub(instances.start)
                        .ok_or(Error::InvalidRange)?,
                ),
                number(indices.start),
                number(base_vertex),
                number(instances.start),
            ],
        )
        .map_err(error)?;
        Ok(())
    }

    fn end_raster(encoder: &mut WebGpuEncoder) -> Result<()> {
        if encoder.occlusion_active {
            return Err(Error::InvalidState);
        }
        let pass = encoder.pass.take().ok_or(Error::InvalidState)?;
        js::call0(&pass, "end").map_err(error)?;
        super::query_segments::flush_pending(encoder)?;
        encoder.occlusion_query_set = None;
        encoder.occlusion_logical_set = None;
        encoder.render_area = None;
        for pending in std::mem::take(&mut encoder.partial_resolves) {
            super::partial::finish_resolve(encoder, pending)?;
        }
        if let Some((set, slot)) = encoder.partial_timestamp_end.take() {
            <WebGpu as fluxel_rhi_core::backend::QueryBackend>::write_timestamp(
                encoder, &set, slot,
            )?;
        }
        Ok(())
    }

    fn abort_raster(encoder: &mut WebGpuEncoder) {
        if let Some(pass) = encoder.pass.take() {
            let _ = js::call0(&pass, "end");
        }
        encoder.occlusion_query_set = None;
        encoder.occlusion_logical_set = None;
        encoder.occlusion_active = false;
        encoder.render_area = None;
        encoder.partial_resolves.clear();
        encoder.partial_timestamp_end = None;
    }
}
