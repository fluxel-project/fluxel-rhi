//! Native WebGPU command encoder declarations.
//!
//! Browser values are retained by the owner-thread registry.  Encoder and
//! command-buffer values therefore carry only opaque registry IDs and remain
//! `Send` even though `JsValue` itself is not.

use std::any::Any;
use std::collections::BTreeMap;

use js_sys::{Function, Object, Reflect};
use wasm_bindgen::{JsCast, JsValue};

use crate::api::binding::{BindGroup, BindGroupIndex};
use crate::api::command::backend::{CommandBufferBackend, CommandEncoderBackend};
use crate::api::command::copy::{BufferCopy, BufferTextureCopy, TextureCopy};
use crate::api::command::geometry::{Color, Rect, Viewport};
use crate::api::command::record::{ComputeBegin, ImmediateWrite, RasterBegin};
use crate::api::command::{IndexFormat, RasterAttachmentClear, ResourceUse, SecondaryRasterWork};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::pipeline::{ComputePipeline, RasterPipeline};
use crate::api::resource::buffer::{Buffer, BufferBinding, BufferRange};
use crate::api::resource::transfer::{ReadbackTicket, UploadDescriptor, UploadJob};

use super::binding::WebGpuBindGroup;
use super::pipeline::{WebGpuComputePipeline, WebGpuRasterPipeline};
use super::registry::{self, WebGpuDriver, WebGpuObjectId, WebGpuRegistration};
use super::resource::{WebGpuBuffer, WebGpuTexture};

fn fail(at: &'static str, message: impl Into<String>) -> RhiError {
    RhiError::new(RhiErrorKind::BackendFailure, message.into()).at(at)
}
fn unsupported(what: &'static str) -> RhiError {
    RhiError::new(
        RhiErrorKind::Unsupported,
        format!("WebGPU native encoder does not support {what}"),
    )
}
fn number(value: u64) -> JsValue {
    JsValue::from_f64(value as f64)
}
fn call(receiver: &JsValue, name: &'static str, args: &[&JsValue]) -> Result<JsValue, String> {
    let function = Reflect::get(receiver, &JsValue::from_str(name))
        .map_err(|e| format!("{e:?}"))?
        .dyn_into::<Function>()
        .map_err(|e| format!("{e:?}"))?;
    match args {
        [] => function.call0(receiver),
        [a] => function.call1(receiver, a),
        [a, b] => function.call2(receiver, a, b),
        [a, b, c] => function.call3(receiver, a, b, c),
        [a, b, c, d] => function.call4(receiver, a, b, c, d),
        [a, b, c, d, e] => function.call5(receiver, a, b, c, d, e),
        [a, b, c, d, e, f] => function.call6(receiver, a, b, c, d, e, f),
        _ => return Err("too many WebGPU call arguments".into()),
    }
    .map_err(|e| format!("{e:?}"))
}
fn get(
    registration: WebGpuRegistration,
    id: WebGpuObjectId,
    what: &'static str,
) -> RhiResult<JsValue> {
    registry::with_object(registration, id, Clone::clone)
        .ok_or_else(|| fail("WebGpuNativeEncoder", format!("{what} was retired")))
}
fn buffer(
    registration: WebGpuRegistration,
    value: &Buffer,
    what: &'static str,
) -> RhiResult<JsValue> {
    let native = value
        .native()
        .as_any()
        .downcast_ref::<WebGpuBuffer>()
        .ok_or_else(|| unsupported(what))?;
    if native.registration() != registration {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            format!("{what} belongs to another WebGPU device"),
        ));
    }
    get(registration, native.object(), what)
}
fn texture(
    registration: WebGpuRegistration,
    value: &crate::api::resource::Texture,
) -> RhiResult<JsValue> {
    let native = value
        .native()
        .as_any()
        .downcast_ref::<WebGpuTexture>()
        .ok_or_else(|| unsupported("texture upload destination"))?;
    if native.registration() != registration {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "texture upload destination belongs to another WebGPU device",
        ));
    }
    get(registration, native.object(), "texture upload destination")
}
fn raster_pipeline(registration: WebGpuRegistration, value: &RasterPipeline) -> RhiResult<JsValue> {
    let native = value
        .native()
        .as_any()
        .downcast_ref::<WebGpuRasterPipeline>()
        .ok_or_else(|| unsupported("raster pipeline"))?;
    if native.registration() != registration {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "raster pipeline belongs to another WebGPU device",
        ));
    }
    get(registration, native.object(), "raster pipeline")
}
fn compute_pipeline(
    registration: WebGpuRegistration,
    value: &ComputePipeline,
) -> RhiResult<JsValue> {
    let native = value
        .native()
        .as_any()
        .downcast_ref::<WebGpuComputePipeline>()
        .ok_or_else(|| unsupported("compute pipeline"))?;
    if native.registration() != registration {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "compute pipeline belongs to another WebGPU device",
        ));
    }
    get(registration, native.object(), "compute pipeline")
}
fn bind_group(registration: WebGpuRegistration, value: &BindGroup) -> RhiResult<JsValue> {
    let native = value
        .native()
        .as_any()
        .downcast_ref::<WebGpuBindGroup>()
        .ok_or_else(|| unsupported("bind group"))?;
    if native.registration() != registration {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "bind group belongs to another WebGPU device",
        ));
    }
    get(registration, native.object(), "bind group")
}

/// Finished GPUCommandBuffer, owned by the registry until submission consumes it.
pub(crate) struct WebGpuCommandBuffer {
    registration: WebGpuRegistration,
    object: WebGpuObjectId,
    readbacks: std::sync::Mutex<Vec<super::command::PendingReadback>>,
}
impl WebGpuCommandBuffer {
    pub(crate) const fn registration(&self) -> WebGpuRegistration {
        self.registration
    }
    pub(crate) const fn object(&self) -> WebGpuObjectId {
        self.object
    }
    /// Submission takes the browser buffer out exactly once.
    pub(crate) fn take(&self) -> Option<JsValue> {
        registry::remove_object(self.registration, self.object)
    }
    pub(crate) fn take_readbacks(&self) -> Vec<super::command::PendingReadback> {
        std::mem::take(&mut *self.readbacks.lock().unwrap_or_else(|p| p.into_inner()))
    }
}
impl CommandBufferBackend for WebGpuCommandBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Clone)]
struct RasterState {
    pipeline: Option<RasterPipeline>,
    groups: BTreeMap<u32, (BindGroup, Vec<u32>)>,
    vertices: BTreeMap<u32, BufferBinding>,
    index: Option<(BufferBinding, IndexFormat)>,
}

/// An open GPUCommandEncoder.  It contains IDs only: no browser value crosses
/// the RHI's `Send` boundary.
pub(crate) struct WebGpuNativeEncoder {
    registration: WebGpuRegistration,
    encoder: WebGpuObjectId,
    raster: Option<(WebGpuObjectId, RasterState)>,
    compute: Option<WebGpuObjectId>,
    readbacks: Vec<super::command::PendingReadback>,
}

pub(crate) fn create_native_encoder(
    driver: &WebGpuDriver,
) -> RhiResult<Box<dyn CommandEncoderBackend>> {
    let registration = driver.registration();
    let device =
        registry::with_device_handles(registration, |h| h.device.clone()).ok_or_else(|| {
            fail(
                "WebGpuNativeEncoder::new",
                "device registration was retired",
            )
        })?;
    let descriptor: JsValue = Object::new().into();
    let value = call(&device, "createCommandEncoder", &[&descriptor])
        .map_err(|e| fail("GPUDevice.createCommandEncoder", e))?;
    let encoder = registry::insert_object(registration, value).ok_or_else(|| {
        fail(
            "WebGpuNativeEncoder::new",
            "device registration was retired",
        )
    })?;
    Ok(Box::new(WebGpuNativeEncoder {
        registration,
        encoder,
        raster: None,
        compute: None,
        readbacks: Vec::new(),
    }))
}
impl WebGpuNativeEncoder {
    fn encoder(&self) -> RhiResult<JsValue> {
        get(self.registration, self.encoder, "command encoder")
    }
    fn raster_pass(&self) -> RhiResult<JsValue> {
        self.raster
            .as_ref()
            .map(|(id, _)| get(self.registration, *id, "render pass"))
            .transpose()?
            .ok_or_else(|| {
                fail(
                    "WebGpuNativeEncoder",
                    "raster operation outside raster scope",
                )
            })
    }
    fn compute_pass(&self) -> RhiResult<JsValue> {
        self.compute
            .map(|id| get(self.registration, id, "compute pass"))
            .transpose()?
            .ok_or_else(|| {
                fail(
                    "WebGpuNativeEncoder",
                    "compute operation outside compute scope",
                )
            })
    }
    fn set_raster_state(&self, pass: &JsValue, state: &RasterState) -> RhiResult<()> {
        let pipeline = state
            .pipeline
            .as_ref()
            .ok_or_else(|| fail("WebGpuNativeEncoder", "draw without raster pipeline"))?;
        let pipeline = raster_pipeline(self.registration, pipeline)?;
        call(pass, "setPipeline", &[&pipeline])
            .map_err(|e| fail("GPURenderPassEncoder.setPipeline", e))?;
        for (index, (group, offsets)) in &state.groups {
            let group = bind_group(self.registration, group)?;
            let offsets_js = js_sys::Array::new();
            for offset in offsets {
                offsets_js.push(&number(*offset as u64));
            }
            let offsets: JsValue = offsets_js.into();
            let index = number(*index as u64);
            call(pass, "setBindGroup", &[&index, &group, &offsets])
                .map_err(|e| fail("GPURenderPassEncoder.setBindGroup", e))?;
        }
        for (slot, binding) in &state.vertices {
            let b = buffer(self.registration, &binding.buffer, "vertex buffer")?;
            let args = [
                number(*slot as u64),
                b,
                number(binding.range.offset),
                number(binding.range.size),
            ];
            call(
                pass,
                "setVertexBuffer",
                &[&args[0], &args[1], &args[2], &args[3]],
            )
            .map_err(|e| fail("GPURenderPassEncoder.setVertexBuffer", e))?;
        }
        if let Some((binding, format)) = &state.index {
            let b = buffer(self.registration, &binding.buffer, "index buffer")?;
            let format = JsValue::from_str(match format {
                IndexFormat::Uint16 => "uint16",
                IndexFormat::Uint32 => "uint32",
            });
            let args = [
                b,
                format,
                number(binding.range.offset),
                number(binding.range.size),
            ];
            call(
                pass,
                "setIndexBuffer",
                &[&args[0], &args[1], &args[2], &args[3]],
            )
            .map_err(|e| fail("GPURenderPassEncoder.setIndexBuffer", e))?;
        }
        Ok(())
    }
    fn end_pass(&self, id: WebGpuObjectId) -> RhiResult<()> {
        let pass = get(self.registration, id, "pass encoder")?;
        call(&pass, "end", &[]).map_err(|e| fail("GPU*PassEncoder.end", e))?;
        let _ = registry::remove_object(self.registration, id);
        Ok(())
    }
    fn stage_texture_upload(&mut self, upload: &UploadJob) -> RhiResult<()> {
        use crate::api::resource::{TextureAspect, TextureDimension};
        let UploadDescriptor::Texture(value) = upload.descriptor() else {
            unreachable!()
        };
        let format = value.dst.descriptor().format;
        let Some(block_bytes) = crate::api::format::logical_bytes_per_block(format) else {
            return Err(unsupported("texture upload with backend-sized texels"));
        };
        let (block_width, block_height) = crate::api::format::block_extent(format);
        let rows = value.extent.height.div_ceil(block_height) as usize;
        let logical_row = (value.extent.width.div_ceil(block_width) as usize)
            .checked_mul(block_bytes as usize)
            .ok_or_else(|| fail("WebGPU texture upload", "row size overflows"))?;
        let pitch = logical_row
            .checked_add(255)
            .map(|n| n & !255)
            .ok_or_else(|| fail("WebGPU texture upload", "row pitch overflows"))?;
        let images = match value.dst.descriptor().dimension {
            TextureDimension::D3 => value.extent.depth,
            _ => value.subresource.layer_count,
        } as usize;
        let mut bytes =
            vec![
                0;
                images
                    .checked_mul(rows)
                    .and_then(|n| n.checked_mul(pitch))
                    .ok_or_else(|| fail("WebGPU texture upload", "staging size overflows"))?
            ];
        for image in 0..images {
            for row in 0..rows {
                let source = image
                    .checked_mul(value.source_layout.rows_per_image as usize)
                    .and_then(|n| n.checked_add(row))
                    .and_then(|n| n.checked_mul(value.source_layout.bytes_per_row as usize))
                    .ok_or_else(|| fail("WebGPU texture upload", "source row offset overflows"))?;
                let destination = image
                    .checked_mul(rows)
                    .and_then(|n| n.checked_add(row))
                    .and_then(|n| n.checked_mul(pitch))
                    .ok_or_else(|| fail("WebGPU texture upload", "staging row offset overflows"))?;
                let source = value
                    .bytes
                    .get(source..source + logical_row)
                    .ok_or_else(|| {
                        fail("WebGPU texture upload", "source bytes do not cover a row")
                    })?;
                bytes[destination..destination + logical_row].copy_from_slice(source);
            }
        }
        let device = registry::with_device_handles(self.registration, |h| h.device.clone())
            .ok_or_else(|| fail("WebGPU texture upload", "device registration was retired"))?;
        let buffer_desc = Object::new();
        Reflect::set(
            &buffer_desc,
            &JsValue::from_str("size"),
            &number(bytes.len() as u64),
        )
        .map_err(|e| fail("GPUBufferDescriptor.size", format!("{e:?}")))?;
        Reflect::set(&buffer_desc, &JsValue::from_str("usage"), &number(4 | 8))
            .map_err(|e| fail("GPUBufferDescriptor.usage", format!("{e:?}")))?;
        Reflect::set(
            &buffer_desc,
            &JsValue::from_str("mappedAtCreation"),
            &JsValue::TRUE,
        )
        .map_err(|e| fail("GPUBufferDescriptor.mappedAtCreation", format!("{e:?}")))?;
        let desc: JsValue = buffer_desc.into();
        let staging = call(&device, "createBuffer", &[&desc])
            .map_err(|e| fail("GPUDevice.createBuffer", e))?;
        let mapped = call(&staging, "getMappedRange", &[])
            .map_err(|e| fail("GPUBuffer.getMappedRange", e))?;
        js_sys::Uint8Array::new(&mapped).set(&js_sys::Uint8Array::from(bytes.as_slice()), 0);
        call(&staging, "unmap", &[]).map_err(|e| fail("GPUBuffer.unmap", e))?;
        let source = Object::new();
        for (key, value) in [
            ("buffer", staging),
            ("offset", number(0)),
            ("bytesPerRow", number(pitch as u64)),
            ("rowsPerImage", number(rows as u64)),
        ] {
            Reflect::set(&source, &JsValue::from_str(key), &value)
                .map_err(|e| fail("GPUImageCopyBuffer", format!("{e:?}")))?;
        }
        let destination = Object::new();
        let texture = texture(self.registration, &value.dst)?;
        Reflect::set(&destination, &JsValue::from_str("texture"), &texture)
            .map_err(|e| fail("GPUImageCopyTexture", format!("{e:?}")))?;
        Reflect::set(
            &destination,
            &JsValue::from_str("mipLevel"),
            &number(value.subresource.mip_level as u64),
        )
        .map_err(|e| fail("GPUImageCopyTexture", format!("{e:?}")))?;
        let origin = Object::new();
        for (key, n) in [
            ("x", value.origin.x),
            ("y", value.origin.y),
            (
                "z",
                value
                    .origin
                    .z
                    .checked_add(value.subresource.base_layer)
                    .ok_or_else(|| fail("WebGPU texture upload", "layer origin overflows"))?,
            ),
        ] {
            Reflect::set(&origin, &JsValue::from_str(key), &number(n as u64))
                .map_err(|e| fail("GPUOrigin3D", format!("{e:?}")))?;
        }
        Reflect::set(&destination, &JsValue::from_str("origin"), &origin)
            .map_err(|e| fail("GPUImageCopyTexture", format!("{e:?}")))?;
        let aspect = match value.subresource.aspect {
            TextureAspect::Color => "all",
            TextureAspect::Depth => "depth-only",
            TextureAspect::Stencil => "stencil-only",
            _ => return Err(unsupported("multi-planar texture upload")),
        };
        Reflect::set(
            &destination,
            &JsValue::from_str("aspect"),
            &JsValue::from_str(aspect),
        )
        .map_err(|e| fail("GPUImageCopyTexture", format!("{e:?}")))?;
        let extent = Object::new();
        for (key, n) in [
            ("width", value.extent.width),
            ("height", value.extent.height),
            ("depthOrArrayLayers", images as u32),
        ] {
            Reflect::set(&extent, &JsValue::from_str(key), &number(n as u64))
                .map_err(|e| fail("GPUExtent3D", format!("{e:?}")))?;
        }
        let source: JsValue = source.into();
        let destination: JsValue = destination.into();
        let extent: JsValue = extent.into();
        let encoder = self.encoder()?;
        call(
            &encoder,
            "copyBufferToTexture",
            &[&source, &destination, &extent],
        )
        .map_err(|e| fail("GPUCommandEncoder.copyBufferToTexture", e))?;
        Ok(())
    }
}

impl CommandEncoderBackend for WebGpuNativeEncoder {
    fn raster_begin(&mut self, begin: &RasterBegin, _: &[ResourceUse]) -> RhiResult<()> {
        if self.raster.is_some() || self.compute.is_some() {
            return Err(fail(
                "WebGpuNativeEncoder::raster_begin",
                "nested command scope",
            ));
        }
        let encoder = self.encoder()?;
        let pass = super::command::native_begin_raster(&encoder, begin, self.registration)?;
        let pass = registry::insert_object(self.registration, pass).ok_or_else(|| {
            fail(
                "WebGpuNativeEncoder::raster_begin",
                "device registration was retired",
            )
        })?;
        self.raster = Some((
            pass,
            RasterState {
                pipeline: None,
                groups: BTreeMap::new(),
                vertices: BTreeMap::new(),
                index: None,
            },
        ));
        Ok(())
    }
    fn raster_set_pipeline(&mut self, p: &RasterPipeline) -> RhiResult<()> {
        self.raster
            .as_mut()
            .ok_or_else(|| {
                fail(
                    "WebGpuNativeEncoder",
                    "raster operation outside raster scope",
                )
            })?
            .1
            .pipeline = Some(p.clone());
        Ok(())
    }
    fn raster_set_bind_group(
        &mut self,
        i: BindGroupIndex,
        g: &BindGroup,
        o: &[u32],
    ) -> RhiResult<()> {
        self.raster
            .as_mut()
            .ok_or_else(|| {
                fail(
                    "WebGpuNativeEncoder",
                    "raster operation outside raster scope",
                )
            })?
            .1
            .groups
            .insert(i.get(), (g.clone(), o.to_vec()));
        Ok(())
    }
    fn raster_set_vertex_buffer(&mut self, s: u32, b: &BufferBinding) -> RhiResult<()> {
        self.raster
            .as_mut()
            .ok_or_else(|| {
                fail(
                    "WebGpuNativeEncoder",
                    "raster operation outside raster scope",
                )
            })?
            .1
            .vertices
            .insert(s, b.clone());
        Ok(())
    }
    fn raster_set_index_buffer(&mut self, b: &BufferBinding, f: IndexFormat) -> RhiResult<()> {
        self.raster
            .as_mut()
            .ok_or_else(|| {
                fail(
                    "WebGpuNativeEncoder",
                    "raster operation outside raster scope",
                )
            })?
            .1
            .index = Some((b.clone(), f));
        Ok(())
    }
    fn raster_set_viewport(&mut self, v: Viewport) -> RhiResult<()> {
        let p = self.raster_pass()?;
        let a = [
            JsValue::from_f64(v.x as f64),
            JsValue::from_f64(v.y as f64),
            JsValue::from_f64(v.width as f64),
            JsValue::from_f64(v.height as f64),
            JsValue::from_f64(v.min_depth as f64),
            JsValue::from_f64(v.max_depth as f64),
        ];
        call(
            &p,
            "setViewport",
            &[&a[0], &a[1], &a[2], &a[3], &a[4], &a[5]],
        )
        .map_err(|e| fail("GPURenderPassEncoder.setViewport", e))?;
        Ok(())
    }
    fn raster_set_scissor(&mut self, r: Rect) -> RhiResult<()> {
        let p = self.raster_pass()?;
        let a = [
            number(r.x as u64),
            number(r.y as u64),
            number(r.width as u64),
            number(r.height as u64),
        ];
        call(&p, "setScissorRect", &[&a[0], &a[1], &a[2], &a[3]])
            .map_err(|e| fail("GPURenderPassEncoder.setScissorRect", e))?;
        Ok(())
    }
    fn raster_set_blend_constant(&mut self, c: Color) -> RhiResult<()> {
        let p = self.raster_pass()?;
        let v = Object::new();
        for (n, x) in [("r", c.r), ("g", c.g), ("b", c.b), ("a", c.a)] {
            Reflect::set(&v, &JsValue::from_str(n), &JsValue::from_f64(x as f64))
                .map_err(|e| fail("blend constant", format!("{e:?}")))?;
        }
        let v: JsValue = v.into();
        call(&p, "setBlendConstant", &[&v])
            .map_err(|e| fail("GPURenderPassEncoder.setBlendConstant", e))?;
        Ok(())
    }
    fn raster_set_stencil_reference(&mut self, v: u32) -> RhiResult<()> {
        let p = self.raster_pass()?;
        let v = number(v as u64);
        call(&p, "setStencilReference", &[&v])
            .map_err(|e| fail("GPURenderPassEncoder.setStencilReference", e))?;
        Ok(())
    }
    fn raster_set_immediates(&mut self, _: &ImmediateWrite) -> RhiResult<()> {
        Err(unsupported("push constants"))
    }
    fn raster_draw(
        &mut self,
        v: core::ops::Range<u32>,
        i: core::ops::Range<u32>,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        let p = self.raster_pass()?;
        self.set_raster_state(&p, &self.raster.as_ref().unwrap().1)?;
        let a = [
            number((v.end - v.start) as u64),
            number((i.end - i.start) as u64),
            number(v.start as u64),
            number(i.start as u64),
        ];
        call(&p, "draw", &[&a[0], &a[1], &a[2], &a[3]])
            .map_err(|e| fail("GPURenderPassEncoder.draw", e))?;
        Ok(())
    }
    fn raster_draw_indexed(
        &mut self,
        v: core::ops::Range<u32>,
        base: i32,
        i: core::ops::Range<u32>,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        let p = self.raster_pass()?;
        let s = &self.raster.as_ref().unwrap().1;
        if s.index.is_none() {
            return Err(fail(
                "WebGpuNativeEncoder",
                "indexed draw without index buffer",
            ));
        }
        self.set_raster_state(&p, s)?;
        let a = [
            number((v.end - v.start) as u64),
            number((i.end - i.start) as u64),
            number(v.start as u64),
            JsValue::from_f64(base as f64),
            number(i.start as u64),
        ];
        call(&p, "drawIndexed", &[&a[0], &a[1], &a[2], &a[3], &a[4]])
            .map_err(|e| fail("GPURenderPassEncoder.drawIndexed", e))?;
        Ok(())
    }
    fn raster_end(&mut self) -> RhiResult<()> {
        let (id, _) = self
            .raster
            .take()
            .ok_or_else(|| fail("WebGpuNativeEncoder", "raster end without begin"))?;
        self.end_pass(id)
    }
    fn raster_clear(&mut self, _: &RasterAttachmentClear, _: &[ResourceUse]) -> RhiResult<()> {
        Err(unsupported("mid-pass attachment clear"))
    }
    fn raster_execute_secondary(
        &mut self,
        _: SecondaryRasterWork,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        Err(unsupported("render bundles"))
    }
    fn compute_begin(&mut self, _: &ComputeBegin) -> RhiResult<()> {
        if self.raster.is_some() || self.compute.is_some() {
            return Err(fail(
                "WebGpuNativeEncoder::compute_begin",
                "nested command scope",
            ));
        }
        let e = self.encoder()?;
        let d: JsValue = Object::new().into();
        let v = call(&e, "beginComputePass", &[&d])
            .map_err(|e| fail("GPUCommandEncoder.beginComputePass", e))?;
        self.compute = Some(
            registry::insert_object(self.registration, v).ok_or_else(|| {
                fail(
                    "WebGpuNativeEncoder::compute_begin",
                    "device registration was retired",
                )
            })?,
        );
        Ok(())
    }
    fn compute_set_pipeline(&mut self, p: &ComputePipeline) -> RhiResult<()> {
        let pass = self.compute_pass()?;
        let p = compute_pipeline(self.registration, p)?;
        call(&pass, "setPipeline", &[&p])
            .map_err(|e| fail("GPUComputePassEncoder.setPipeline", e))?;
        Ok(())
    }
    fn compute_set_bind_group(
        &mut self,
        i: BindGroupIndex,
        g: &BindGroup,
        o: &[u32],
    ) -> RhiResult<()> {
        let p = self.compute_pass()?;
        let g = bind_group(self.registration, g)?;
        let a = js_sys::Array::new();
        for x in o {
            a.push(&number(*x as u64));
        }
        let a: JsValue = a.into();
        let i = number(i.get() as u64);
        call(&p, "setBindGroup", &[&i, &g, &a])
            .map_err(|e| fail("GPUComputePassEncoder.setBindGroup", e))?;
        Ok(())
    }
    fn compute_set_immediates(&mut self, _: &ImmediateWrite) -> RhiResult<()> {
        Err(unsupported("push constants"))
    }
    fn compute_dispatch(&mut self, x: u32, y: u32, z: u32, _: &[ResourceUse]) -> RhiResult<()> {
        let p = self.compute_pass()?;
        let a = [number(x as u64), number(y as u64), number(z as u64)];
        call(&p, "dispatchWorkgroups", &[&a[0], &a[1], &a[2]])
            .map_err(|e| fail("GPUComputePassEncoder.dispatchWorkgroups", e))?;
        Ok(())
    }
    fn compute_dispatch_indirect(
        &mut self,
        b: &Buffer,
        o: u64,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        let p = self.compute_pass()?;
        let b = buffer(self.registration, b, "compute indirect arguments")?;
        let o = number(o);
        call(&p, "dispatchWorkgroupsIndirect", &[&b, &o])
            .map_err(|e| fail("GPUComputePassEncoder.dispatchWorkgroupsIndirect", e))?;
        Ok(())
    }
    fn compute_end(&mut self) -> RhiResult<()> {
        let id = self
            .compute
            .take()
            .ok_or_else(|| fail("WebGpuNativeEncoder", "compute end without begin"))?;
        self.end_pass(id)
    }
    fn copy_buffer(&mut self, c: &BufferCopy, _: &[ResourceUse]) -> RhiResult<()> {
        let e = self.encoder()?;
        let a = [
            buffer(self.registration, &c.src, "copy source")?,
            number(c.src_offset),
            buffer(self.registration, &c.dst, "copy destination")?,
            number(c.dst_offset),
            number(c.size),
        ];
        call(
            &e,
            "copyBufferToBuffer",
            &[&a[0], &a[1], &a[2], &a[3], &a[4]],
        )
        .map_err(|e| fail("GPUCommandEncoder.copyBufferToBuffer", e))?;
        Ok(())
    }
    fn clear_buffer(&mut self, b: &Buffer, r: BufferRange, _: &[ResourceUse]) -> RhiResult<()> {
        let e = self.encoder()?;
        let a = [
            buffer(self.registration, b, "clear buffer")?,
            number(r.offset),
            number(r.size),
        ];
        call(&e, "clearBuffer", &[&a[0], &a[1], &a[2]])
            .map_err(|e| fail("GPUCommandEncoder.clearBuffer", e))?;
        Ok(())
    }
    fn copy_buffer_to_texture(
        &mut self,
        c: &BufferTextureCopy,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        let e = self.encoder()?;
        super::command::lower_buffer_texture_copy(&e, c, true)
            .map_err(|e| fail("GPUCommandEncoder.copyBufferToTexture", e))
    }
    fn copy_texture_to_buffer(
        &mut self,
        c: &BufferTextureCopy,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        let e = self.encoder()?;
        super::command::lower_buffer_texture_copy(&e, c, false)
            .map_err(|e| fail("GPUCommandEncoder.copyTextureToBuffer", e))
    }
    fn copy_texture(&mut self, c: &TextureCopy, _: &[ResourceUse]) -> RhiResult<()> {
        let e = self.encoder()?;
        super::command::lower_texture_copy(&e, c)
            .map_err(|e| fail("GPUCommandEncoder.copyTextureToTexture", e))
    }
    fn encode_upload(&mut self, upload: &UploadJob, _: &[ResourceUse]) -> RhiResult<()> {
        let UploadDescriptor::Buffer(value) = upload.descriptor() else {
            return self.stage_texture_upload(upload);
        };
        if value.bytes.is_empty() {
            return Ok(());
        }
        // copyBufferToBuffer cannot express byte-granular writes.  Padding the
        // copy would overwrite destination bytes the upload does not own.
        if !value.dst_offset.is_multiple_of(4) || !value.bytes.len().is_multiple_of(4) {
            return Err(unsupported(
                "unaligned buffer upload for native copy staging",
            ));
        }
        // GPUQueue.writeBuffer submits outside this encoder.  A mapped staging
        // buffer keeps upload ordering inside this native command buffer.
        let device = registry::with_device_handles(self.registration, |h| h.device.clone())
            .ok_or_else(|| {
                fail(
                    "WebGpuNativeEncoder::encode_upload",
                    "device registration was retired",
                )
            })?;
        let descriptor = Object::new();
        Reflect::set(
            &descriptor,
            &JsValue::from_str("size"),
            &number(value.bytes.len() as u64),
        )
        .map_err(|e| fail("GPUBufferDescriptor.size", format!("{e:?}")))?;
        Reflect::set(&descriptor, &JsValue::from_str("usage"), &number(4 | 8))
            .map_err(|e| fail("GPUBufferDescriptor.usage", format!("{e:?}")))?;
        Reflect::set(
            &descriptor,
            &JsValue::from_str("mappedAtCreation"),
            &JsValue::TRUE,
        )
        .map_err(|e| fail("GPUBufferDescriptor.mappedAtCreation", format!("{e:?}")))?;
        let descriptor: JsValue = descriptor.into();
        let staging = call(&device, "createBuffer", &[&descriptor])
            .map_err(|e| fail("GPUDevice.createBuffer", e))?;
        let mapped = call(&staging, "getMappedRange", &[])
            .map_err(|e| fail("GPUBuffer.getMappedRange", e))?;
        js_sys::Uint8Array::new(&mapped).set(&js_sys::Uint8Array::from(value.bytes.as_ref()), 0);
        call(&staging, "unmap", &[]).map_err(|e| fail("GPUBuffer.unmap", e))?;
        let destination = buffer(self.registration, &value.dst, "upload destination")?;
        let encoder = self.encoder()?;
        let args = [
            staging,
            number(0),
            destination,
            number(value.dst_offset),
            number(value.bytes.len() as u64),
        ];
        call(
            &encoder,
            "copyBufferToBuffer",
            &[&args[0], &args[1], &args[2], &args[3], &args[4]],
        )
        .map_err(|e| fail("GPUCommandEncoder.copyBufferToBuffer", e))?;
        Ok(())
    }
    fn encode_readback(&mut self, ticket: &ReadbackTicket, _: &[ResourceUse]) -> RhiResult<()> {
        let encoder = self.encoder()?;
        super::command::lower_readback(&encoder, ticket, self.registration, &mut self.readbacks)
            .map_err(|e| fail("WebGPU readback", e))
    }
    fn finish(mut self: Box<Self>) -> RhiResult<Box<dyn CommandBufferBackend>> {
        if self.raster.is_some() || self.compute.is_some() {
            return Err(fail("WebGpuNativeEncoder::finish", "scope still open"));
        }
        let e = self.encoder()?;
        let v = call(&e, "finish", &[]).map_err(|e| fail("GPUCommandEncoder.finish", e))?;
        let _ = registry::remove_object(self.registration, self.encoder);
        let object = registry::insert_object(self.registration, v).ok_or_else(|| {
            fail(
                "WebGpuNativeEncoder::finish",
                "device registration was retired",
            )
        })?;
        Ok(Box::new(WebGpuCommandBuffer {
            registration: self.registration,
            object,
            readbacks: std::sync::Mutex::new(std::mem::take(&mut self.readbacks)),
        }))
    }
}
