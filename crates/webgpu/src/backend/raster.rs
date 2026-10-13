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

impl api::RasterBackend for WebGpu {
    fn begin_raster(
        encoder: &mut WebGpuEncoder,
        desc: &api::RasterPassDesc<'_, Self>,
    ) -> Result<()> {
        if encoder.pass.is_some()
            || desc.multiview_mask.is_some()
            || desc.timestamp_writes.is_some()
            || desc.secondary_inheritance.is_some()
        {
            return Err(Error::Unsupported("WebGPU raster pass option"));
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
                field(&attachment, "resolveTarget", target.clone())?;
                encoder.keep_alive.push(target);
            }
            let load = match color.load {
                api::LoadOp::Load => "load",
                api::LoadOp::Clear(_) | api::LoadOp::Discard => "clear",
            };
            field(&attachment, "loadOp", JsValue::from_str(load))?;
            if let api::LoadOp::Clear(clear) = color.load {
                let api::ClearColor::Float(rgba) = clear else {
                    return Err(Error::Unsupported("WebGPU integer clear color"));
                };
                let value = object();
                for (channel, component) in ["r", "g", "b", "a"].into_iter().zip(rgba) {
                    field(&value, channel, number(component as f64))?;
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
        if let Some(set) = desc.occlusion_query_set {
            // SAFETY: `begin_raster` checked the query set DeviceId; this clone
            // is retained by the native encoder until queue completion.
            let raw = unsafe { set.native_ref() }.raw.clone();
            field(&native, "occlusionQuerySet", raw.clone())?;
            encoder.keep_alive.push(raw);
        }
        if let Some(depth) = &desc.depth_stencil {
            if depth.stencil.is_some() {
                return Err(Error::Unsupported("WebGPU stencil attachment"));
            }
            let Some((load, store)) = depth.depth else {
                return Err(Error::Unsupported(
                    "WebGPU depth attachment without depth operations",
                ));
            };
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
            field(
                &attachment,
                "depthLoadOp",
                JsValue::from_str(match load {
                    api::LoadOp::Load => "load",
                    api::LoadOp::Clear(_) | api::LoadOp::Discard => "clear",
                }),
            )?;
            if let api::LoadOp::Clear(value) = load {
                field(&attachment, "depthClearValue", number(value as f64))?;
            } else if matches!(load, api::LoadOp::Discard) {
                field(&attachment, "depthClearValue", number(1.0_f64))?;
            }
            field(
                &attachment,
                "depthStoreOp",
                JsValue::from_str(match store {
                    api::StoreOp::Store => "store",
                    api::StoreOp::Discard => "discard",
                }),
            )?;
            field(
                &attachment,
                "depthReadOnly",
                JsValue::from_bool(depth.depth_read_only),
            )?;
            field(&native, "depthStencilAttachment", attachment)?;
        }
        encoder.pass = Some(js::call1(&encoder.raw, "beginRenderPass", &native).map_err(error)?);
        encoder.occlusion_query_set = desc.occlusion_query_set.map(|set| {
            // SAFETY: `begin_raster` checked the query set DeviceId; this clone
            // only identifies the current pass until `end_raster` clears it.
            unsafe { set.native_ref() }.raw.clone()
        });
        Ok(())
    }

    fn set_pipeline(encoder: &mut WebGpuEncoder, pipeline: &JsValue) -> Result<()> {
        js::call1(pass(encoder)?, "setPipeline", pipeline).map_err(error)?;
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
            &[number(slot), group.clone(), dynamic.into()],
        )
        .map_err(error)?;
        encoder.keep_alive.push(group.clone());
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
        Ok(())
    }

    fn scissor(encoder: &mut WebGpuEncoder, rect: api::Rect) -> Result<()> {
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
        Ok(())
    }

    fn blend_constant(encoder: &mut WebGpuEncoder, value: [f32; 4]) -> Result<()> {
        let color = object();
        for (name, component) in ["r", "g", "b", "a"].into_iter().zip(value) {
            field(&color, name, number(component as f64))?;
        }
        js::call1(pass(encoder)?, "setBlendConstant", &color).map_err(error)?;
        Ok(())
    }

    fn stencil_reference(encoder: &mut WebGpuEncoder, value: u32) -> Result<()> {
        js::call1(pass(encoder)?, "setStencilReference", &number(value)).map_err(error)?;
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
        encoder.occlusion_query_set = None;
        Ok(())
    }

    fn abort_raster(encoder: &mut WebGpuEncoder) {
        if let Some(pass) = encoder.pass.take() {
            let _ = js::call0(&pass, "end");
        }
        encoder.occlusion_query_set = None;
        encoder.occlusion_active = false;
    }
}
