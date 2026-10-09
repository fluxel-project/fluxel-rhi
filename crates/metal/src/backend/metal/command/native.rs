//! Immediate Metal command encoding.
//!
//! The portable encoder validates calls and tracks resource use; this module
//! lowers each accepted call into the recorder's `MTLCommandBuffer` immediately.
//! A finished token owns that buffer and its staging allocations so submission
//! only commits native work.

use std::any::Any;
use std::sync::{Arc, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLRenderCommandEncoder, MTLResourceOptions,
    MTLVisibilityResultMode,
};

use crate::api::binding::{BindGroup, BindGroupIndex};
use crate::api::command::backend::{CommandBufferBackend, CommandEncoderBackend};
use crate::api::command::copy::{
    BufferCopy, BufferTextureCopy, TextureBlit, TextureCopy, TextureResolve,
};
use crate::api::command::record::{
    BoundGroup, BoundIndexBuffer, ComputeBegin, ImmediateWrite, RasterBegin,
};
use crate::api::command::{Color, IndexFormat, Rect, ResourceUse, Viewport};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::pipeline::{ComputePipeline, RasterPipeline};
use crate::api::query::QuerySet;
use crate::api::resource::Texture;
use crate::api::resource::buffer::{Buffer, BufferBinding, BufferRange};
use crate::api::resource::subresource::TextureSubresourceRange;
use crate::api::resource::transfer::{ReadbackTicket, UploadDescriptor, UploadJob};

use super::super::device::MetalShared;
use super::{
    MetalPendingReadback, bind_compute_groups, bind_compute_immediates, bind_raster_groups,
    bind_raster_immediates, bind_vertex_buffers, end_blit, ensure_blit, lower_clear_texture,
    lower_readback, metal_buffer, metal_cull_mode, metal_fill_mode, metal_index_type,
    metal_primitive, metal_scissor, metal_texture, metal_viewport, metal_winding, origin,
    raster_scope_extent, render_pass_descriptor, repack_texture_upload, size,
    validate_base_vertex_instance_selector,
};

/// One opened Metal command buffer, owned by a single portable recorder.
pub(crate) struct MetalNativeEncoder {
    shared: Arc<MetalShared>,
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    blit: Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
    compute: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    render: Option<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>>,
    raster_extent: Option<(u32, u32)>,
    raster: TypedRasterState,
    compute_state: TypedComputeState,
    readbacks: Vec<MetalPendingReadback>,
    // Metal command encoders retain their bound resources, but these buffers
    // are created solely for uploads. Keep an explicit ownership edge until
    // submission takes the finished token.
    retained_buffers: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    visibility_scratch: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    visibility_next: usize,
    active_query: Option<(QuerySet, u32, usize)>,
    pending_queries: Vec<(QuerySet, u32, usize)>,
}

// Objective-C Metal objects are reference-counted and the command spine
// serializes queue submission. Their Rust protocol wrappers deliberately do
// not claim Send, while the portable command-buffer seam requires it.
unsafe impl Send for MetalNativeEncoder {}

struct TypedRasterState {
    pipeline: Option<RasterPipeline>,
    groups: Vec<BoundGroup>,
    vertex_buffers: Vec<(u32, BufferBinding)>,
    index: Option<BoundIndexBuffer>,
    viewport: Option<Viewport>,
    scissor: Option<Rect>,
    blend: Color,
    stencil: u32,
    immediates: Vec<ImmediateWrite>,
}
impl Default for TypedRasterState {
    fn default() -> Self {
        Self {
            pipeline: None,
            groups: Vec::new(),
            vertex_buffers: Vec::new(),
            index: None,
            viewport: None,
            scissor: None,
            blend: Color::new(0.0, 0.0, 0.0, 0.0),
            stencil: 0,
            immediates: Vec::new(),
        }
    }
}
struct TypedComputeState {
    pipeline: Option<ComputePipeline>,
    groups: Vec<BoundGroup>,
    immediates: Vec<ImmediateWrite>,
}
impl Default for TypedComputeState {
    fn default() -> Self {
        Self {
            pipeline: None,
            groups: Vec::new(),
            immediates: Vec::new(),
        }
    }
}

pub(crate) struct MetalNativeCommandBuffer {
    finished: Mutex<Option<FinishedMetalCommandBuffer>>,
}
unsafe impl Send for MetalNativeCommandBuffer {}
pub(super) struct FinishedMetalCommandBuffer {
    pub(super) command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    pub(super) readbacks: Vec<MetalPendingReadback>,
    /// Upload staging must outlive every committed buffer in a plan batch.
    /// Submission moves this into the final completion closure.
    pub(super) retained_staging: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
}
unsafe impl Send for FinishedMetalCommandBuffer {}

impl MetalNativeEncoder {
    pub(crate) fn new(shared: Arc<MetalShared>) -> RhiResult<Self> {
        let command_buffer = shared.queue.commandBuffer().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal failed to allocate command buffer",
            )
            .at("MetalNativeEncoder::new")
        })?;
        let visibility_scratch = shared
            .device
            .newBufferWithLength_options(65_536 * 8, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::OutOfMemory,
                    "Metal visibility scratch allocation failed",
                )
                .at("MetalNativeEncoder::new")
            })?;
        Ok(Self {
            shared,
            command_buffer,
            blit: None,
            compute: None,
            render: None,
            raster_extent: None,
            raster: TypedRasterState::default(),
            compute_state: TypedComputeState::default(),
            readbacks: Vec::new(),
            retained_buffers: Vec::new(),
            visibility_scratch,
            visibility_next: 0,
            active_query: None,
            pending_queries: Vec::new(),
        })
    }
    fn invalid<T>(&self, message: &'static str) -> RhiResult<T> {
        Err(RhiError::new(RhiErrorKind::InvalidUsage, message).at("MetalNativeEncoder"))
    }
    fn end_compute(&mut self) {
        if let Some(encoder) = self.compute.take() {
            encoder.endEncoding();
        }
    }
    fn end_render(&mut self) {
        if let Some(encoder) = self.render.take() {
            encoder.endEncoding();
            self.raster_extent = None;
        }
    }
    fn flush_visibility(&mut self) -> RhiResult<()> {
        if self.active_query.is_some() {
            return self.invalid("Metal raster scope ended with an active occlusion query");
        }
        if self.pending_queries.is_empty() {
            return Ok(());
        }
        let blit = ensure_blit(&self.command_buffer, &mut self.blit)?;
        for (set, index, offset) in self.pending_queries.drain(..) {
            let native = super::metal_query_set(&set)?;
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &self.visibility_scratch,
                    offset,
                    &native.raw,
                    (u64::from(index) * 8) as usize,
                    8,
                );
            }
        }
        Ok(())
    }
    fn debug_push(&mut self, label: &str) {
        let label = NSString::from_str(label);
        if let Some(encoder) = self.render.as_deref() {
            encoder.pushDebugGroup(&label);
        } else if let Some(encoder) = self.compute.as_deref() {
            encoder.pushDebugGroup(&label);
        } else if let Some(encoder) = self.blit.as_deref() {
            encoder.pushDebugGroup(&label);
        }
    }
    fn debug_pop(&mut self) {
        if let Some(encoder) = self.render.as_deref() {
            encoder.popDebugGroup();
        } else if let Some(encoder) = self.compute.as_deref() {
            encoder.popDebugGroup();
        } else if let Some(encoder) = self.blit.as_deref() {
            encoder.popDebugGroup();
        }
    }
    fn debug_marker(&mut self, label: &str) {
        let label = NSString::from_str(label);
        if let Some(encoder) = self.render.as_deref() {
            encoder.insertDebugSignpost(&label);
        } else if let Some(encoder) = self.compute.as_deref() {
            encoder.insertDebugSignpost(&label);
        } else if let Some(encoder) = self.blit.as_deref() {
            encoder.insertDebugSignpost(&label);
        }
    }
    fn ensure_compute(&mut self) -> RhiResult<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        if self.render.is_some() {
            return self.invalid("Metal compute scope overlaps a raster scope");
        }
        end_blit(&mut self.blit);
        if self.compute.is_none() {
            self.compute = Some(self.command_buffer.computeCommandEncoder().ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "Metal failed to create compute encoder",
                )
                .at("MetalNativeEncoder::compute_begin")
            })?);
        }
        Ok(self.compute.as_deref().expect("created above"))
    }
    fn bound<T: Clone>(items: &mut Vec<T>, item: T, same: impl Fn(&T) -> bool) {
        if let Some(old) = items.iter_mut().find(|old| same(old)) {
            *old = item;
        } else {
            items.push(item);
        }
    }
    fn draw(
        &mut self,
        range: core::ops::Range<u32>,
        instances: core::ops::Range<u32>,
        base_vertex: i32,
        indexed: bool,
    ) -> RhiResult<()> {
        let encoder = self.render.as_deref().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal raster draw outside raster scope",
            )
            .at("MetalNativeEncoder::draw")
        })?;
        let pipeline = self.raster.pipeline.as_ref().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal raster draw requires a pipeline",
            )
            .at("MetalNativeEncoder::draw")
        })?;
        let native = pipeline
            .native()
            .as_any()
            .downcast_ref::<super::super::pipeline::MetalRasterPipeline>()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "raster pipeline is not backed by Metal",
                )
                .at("MetalNativeEncoder::draw")
            })?;
        encoder.setRenderPipelineState(native.state());
        if let Some(mappings) = native.vertex_amplification_mappings() {
            unsafe {
                encoder.setVertexAmplificationCount_viewMappings(mappings.len(), mappings.as_ptr());
            }
        }
        if let Some(mode) = native.depth_clip_mode() {
            encoder.setDepthClipMode(mode);
        }
        encoder.setDepthStencilState(native.depth_stencil());
        encoder.setStencilReferenceValue(self.raster.stencil);
        let primitive = &pipeline.descriptor().primitive;
        encoder.setCullMode(metal_cull_mode(primitive.cull_mode));
        encoder.setFrontFacingWinding(metal_winding(primitive.front_face));
        encoder.setTriangleFillMode(metal_fill_mode(primitive.polygon_mode)?);
        encoder.setBlendColorRed_green_blue_alpha(
            self.raster.blend.r,
            self.raster.blend.g,
            self.raster.blend.b,
            self.raster.blend.a,
        );
        let extent = self.raster_extent.ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal raster scope has no extent",
            )
        })?;
        encoder.setViewport(metal_viewport(self.raster.viewport.unwrap_or(
            Viewport::new(0.0, 0.0, extent.0 as f32, extent.1 as f32, 0.0, 1.0),
        )));
        encoder.setScissorRect(metal_scissor(
            self.raster
                .scissor
                .unwrap_or(Rect::new(0, 0, extent.0, extent.1)),
        ));
        bind_vertex_buffers(encoder, &self.raster.vertex_buffers)?;
        bind_raster_groups(encoder, native.binding_abi(), &self.raster.groups)?;
        bind_raster_immediates(encoder, native.binding_abi(), &self.raster.immediates)?;
        let count = range
            .end
            .checked_sub(range.start)
            .ok_or_else(|| RhiError::new(RhiErrorKind::InvalidUsage, "draw range underflows"))?;
        let instance_count = instances.end.checked_sub(instances.start).ok_or_else(|| {
            RhiError::new(RhiErrorKind::InvalidUsage, "instance range underflows")
        })?;
        validate_base_vertex_instance_selector(
            self.shared.base_vertex_instance,
            indexed,
            base_vertex,
            instances.start,
        )?;
        if indexed {
            let index = self.raster.index.as_ref().ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "indexed draw requires index buffer",
                )
            })?;
            let buffer = metal_buffer(&index.binding.buffer)?;
            let offset = index
                .binding
                .range
                .offset
                .checked_add(
                    u64::from(range.start)
                        .checked_mul(super::index_element_size(index.format))
                        .ok_or_else(|| {
                            RhiError::new(RhiErrorKind::InvalidUsage, "index offset overflow")
                        })?,
                )
                .ok_or_else(|| {
                    RhiError::new(RhiErrorKind::InvalidUsage, "index offset overflow")
                })?;
            unsafe {
                if self.shared.base_vertex_instance {
                    encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(metal_primitive(primitive.topology), count as usize, metal_index_type(index.format), &buffer.raw, offset as usize, instance_count as usize, base_vertex as isize, instances.start as usize);
                } else {
                    encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount(metal_primitive(primitive.topology), count as usize, metal_index_type(index.format), &buffer.raw, offset as usize, instance_count as usize);
                }
            }
        } else {
            unsafe {
                if self.shared.base_vertex_instance {
                    encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                        metal_primitive(primitive.topology),
                        range.start as usize,
                        count as usize,
                        instance_count as usize,
                        instances.start as usize,
                    );
                } else {
                    encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                        metal_primitive(primitive.topology),
                        range.start as usize,
                        count as usize,
                        instance_count as usize,
                    );
                }
            }
        }
        Ok(())
    }
}

impl CommandEncoderBackend for MetalNativeEncoder {
    fn clear_buffer(
        &mut self,
        buffer: &Buffer,
        range: BufferRange,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        self.end_compute();
        self.end_render();
        let e = ensure_blit(&self.command_buffer, &mut self.blit)?;
        let b = metal_buffer(buffer)?;
        e.fillBuffer_range_value(
            &b.raw,
            objc2_foundation::NSRange {
                location: range.offset as usize,
                length: range.size as usize,
            },
            0,
        );
        Ok(())
    }
    fn clear_texture(
        &mut self,
        texture: &Texture,
        range: TextureSubresourceRange,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        self.end_compute();
        self.end_render();
        lower_clear_texture(
            &self.shared.device,
            &self.command_buffer,
            &mut self.blit,
            texture,
            range,
        )
    }
    fn copy_buffer(&mut self, copy: &BufferCopy, _: &[ResourceUse]) -> RhiResult<()> {
        self.end_compute();
        self.end_render();
        let e = ensure_blit(&self.command_buffer, &mut self.blit)?;
        let s = metal_buffer(&copy.src)?;
        let d = metal_buffer(&copy.dst)?;
        unsafe {
            e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &s.raw,
                copy.src_offset as usize,
                &d.raw,
                copy.dst_offset as usize,
                copy.size as usize,
            )
        };
        Ok(())
    }
    fn copy_buffer_to_texture(
        &mut self,
        copy: &BufferTextureCopy,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        self.copy_buffer_texture(copy, true)
    }
    fn copy_texture_to_buffer(
        &mut self,
        copy: &BufferTextureCopy,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        self.copy_buffer_texture(copy, false)
    }
    fn copy_texture(&mut self, copy: &TextureCopy, _: &[ResourceUse]) -> RhiResult<()> {
        self.end_compute();
        self.end_render();
        let e = ensure_blit(&self.command_buffer, &mut self.blit)?;
        let s = metal_texture(&copy.src)?;
        let d = metal_texture(&copy.dst)?;
        for layer in 0..copy.src_subresource.layer_count {
            unsafe {
                e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(&s.raw,(copy.src_subresource.base_layer+layer)as usize,copy.src_subresource.mip_level as usize,origin(copy.src_origin),size(copy.extent),&d.raw,(copy.dst_subresource.base_layer+layer)as usize,copy.dst_subresource.mip_level as usize,origin(copy.dst_origin));
            }
        }
        Ok(())
    }
    fn resolve_texture(&mut self, resolve: &TextureResolve, _: &[ResourceUse]) -> RhiResult<()> {
        self.end_compute();
        self.end_render();
        end_blit(&mut self.blit);
        let mut cache = self
            .shared
            .resolve_pipeline
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if cache.is_none() {
            *cache = Some(super::super::resolve::create_pipeline(&self.shared.device)?);
        }
        super::super::resolve::encode(
            &self.command_buffer,
            cache.as_ref().expect("installed above"),
            resolve,
        )
    }
    fn blit_texture(&mut self, _: &TextureBlit, _: &[ResourceUse]) -> RhiResult<()> {
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal baseline does not expose texture blit",
        )
        .at("MetalNativeEncoder::blit_texture"))
    }
    fn encode_upload(&mut self, upload: &UploadJob, _: &[ResourceUse]) -> RhiResult<()> {
        self.end_compute();
        self.end_render();
        match upload.descriptor() {
            UploadDescriptor::Buffer(u) => {
                let d = metal_buffer(&u.dst)?;
                let s = unsafe {
                    self.shared.device.newBufferWithBytes_length_options(
                        std::ptr::NonNull::new(u.bytes.as_ptr() as *mut core::ffi::c_void)
                            .ok_or_else(|| {
                                RhiError::new(RhiErrorKind::InvalidUsage, "empty upload")
                            })?,
                        u.bytes.len(),
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::OutOfMemory,
                        "Metal upload staging allocation failed",
                    )
                })?;
                let e = ensure_blit(&self.command_buffer, &mut self.blit)?;
                unsafe {
                    e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &s,
                        0,
                        &d.raw,
                        u.dst_offset as usize,
                        u.bytes.len(),
                    )
                };
                self.retained_buffers.push(s);
                Ok(())
            }
            UploadDescriptor::Texture(u) => {
                let d = metal_texture(&u.dst)?;
                let packed = repack_texture_upload(u)?;
                let s = unsafe {
                    self.shared.device.newBufferWithBytes_length_options(
                        std::ptr::NonNull::new(packed.bytes.as_ptr() as *mut core::ffi::c_void)
                            .ok_or_else(|| {
                                RhiError::new(RhiErrorKind::InvalidUsage, "empty texture upload")
                            })?,
                        packed.bytes.len(),
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::OutOfMemory,
                        "Metal upload staging allocation failed",
                    )
                })?;
                let e = ensure_blit(&self.command_buffer, &mut self.blit)?;
                for layer in 0..u.subresource.layer_count {
                    unsafe {
                        e.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(&s,(packed.bytes_per_image*u64::from(layer))as usize,packed.bytes_per_row as usize,packed.bytes_per_image as usize,size(u.extent),&d.raw,(u.subresource.base_layer+layer)as usize,u.subresource.mip_level as usize,origin(u.origin));
                    }
                }
                self.retained_buffers.push(s);
                Ok(())
            }
            _ => Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "unknown upload descriptor",
            )),
        }
    }
    fn encode_readback(&mut self, ticket: &ReadbackTicket, _: &[ResourceUse]) -> RhiResult<()> {
        self.end_compute();
        self.end_render();
        lower_readback(
            &self.shared.device,
            &self.command_buffer,
            &mut self.blit,
            ticket,
            &mut self.readbacks,
        )
    }
    fn resolve_query_set(
        &mut self,
        set: &QuerySet,
        first: u32,
        count: u32,
        destination: &Buffer,
        destination_offset: u64,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        self.end_compute();
        self.end_render();
        let source = super::metal_query_set(set)?;
        if first
            .checked_add(count)
            .is_none_or(|end| end > source.count)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "Metal query resolve range exceeds query set",
            ));
        }
        let destination = metal_buffer(destination)?;
        let blit = ensure_blit(&self.command_buffer, &mut self.blit)?;
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &source.raw,
                (u64::from(first) * 8) as usize,
                &destination.raw,
                destination_offset as usize,
                (u64::from(count) * 8) as usize,
            );
        }
        Ok(())
    }
    fn encoder_push_debug_group(&mut self, label: &str) -> RhiResult<()> {
        self.debug_push(label);
        Ok(())
    }
    fn encoder_pop_debug_group(&mut self) -> RhiResult<()> {
        self.debug_pop();
        Ok(())
    }
    fn encoder_insert_debug_marker(&mut self, label: &str) -> RhiResult<()> {
        self.debug_marker(label);
        Ok(())
    }
    fn compute_begin(&mut self, _: &ComputeBegin) -> RhiResult<()> {
        self.ensure_compute().map(|_| ())
    }
    fn compute_set_pipeline(&mut self, p: &ComputePipeline) -> RhiResult<()> {
        self.compute_state.pipeline = Some(p.clone());
        self.compute_state.immediates.clear();
        Ok(())
    }
    fn compute_set_bind_group(
        &mut self,
        index: BindGroupIndex,
        group: &BindGroup,
        offsets: &[u32],
    ) -> RhiResult<()> {
        let v = BoundGroup {
            index,
            group: group.clone(),
            dynamic_offsets: offsets.to_vec(),
        };
        Self::bound(&mut self.compute_state.groups, v, |x| x.index == index);
        Ok(())
    }
    fn compute_set_immediates(&mut self, w: &ImmediateWrite) -> RhiResult<()> {
        let v = w.clone();
        Self::bound(&mut self.compute_state.immediates, v, |x| {
            x.offset == w.offset
        });
        Ok(())
    }
    fn compute_dispatch(&mut self, x: u32, y: u32, z: u32, _: &[ResourceUse]) -> RhiResult<()> {
        let pipeline = self.compute_state.pipeline.clone().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "compute dispatch requires pipeline",
            )
        })?;
        let native = pipeline
            .native()
            .as_any()
            .downcast_ref::<super::super::pipeline::MetalComputePipeline>()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "compute pipeline is not backed by Metal",
                )
            })?;
        let groups = self.compute_state.groups.clone();
        let immediates = self.compute_state.immediates.clone();
        let e = self.ensure_compute()?;
        e.setComputePipelineState(native.state());
        bind_compute_groups(e, native.binding_abi(), &groups)?;
        bind_compute_immediates(e, native.binding_abi(), &immediates)?;
        let local = native.workgroup_size();
        unsafe {
            e.dispatchThreadgroups_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: x as usize,
                    height: y as usize,
                    depth: z as usize,
                },
                objc2_metal::MTLSize {
                    width: local.x as usize,
                    height: local.y as usize,
                    depth: local.z as usize,
                },
            )
        };
        Ok(())
    }
    fn compute_dispatch_indirect(
        &mut self,
        arguments: &Buffer,
        offset: u64,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        let pipeline = self.compute_state.pipeline.clone().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "indirect compute dispatch requires pipeline",
            )
        })?;
        let native = pipeline
            .native()
            .as_any()
            .downcast_ref::<super::super::pipeline::MetalComputePipeline>()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "compute pipeline is not backed by Metal",
                )
            })?;
        let groups = self.compute_state.groups.clone();
        let e = self.ensure_compute()?;
        let arguments = metal_buffer(arguments)?;
        e.setComputePipelineState(native.state());
        bind_compute_groups(e, native.binding_abi(), &groups)?;
        bind_compute_immediates(e, native.binding_abi(), &[])?;
        let local = native.workgroup_size();
        unsafe {
            e.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                &arguments.raw,
                offset as usize,
                objc2_metal::MTLSize {
                    width: local.x as usize,
                    height: local.y as usize,
                    depth: local.z as usize,
                },
            );
        }
        Ok(())
    }
    fn compute_end(&mut self) -> RhiResult<()> {
        if self.compute.is_none() {
            return self.invalid("Metal compute scope end without begin");
        };
        self.end_compute();
        Ok(())
    }
    fn compute_push_debug_group(&mut self, label: &str) -> RhiResult<()> {
        self.debug_push(label);
        Ok(())
    }
    fn compute_pop_debug_group(&mut self) -> RhiResult<()> {
        self.debug_pop();
        Ok(())
    }
    fn compute_insert_debug_marker(&mut self, label: &str) -> RhiResult<()> {
        self.debug_marker(label);
        Ok(())
    }
    fn raster_begin(&mut self, begin: &RasterBegin, _: &[ResourceUse]) -> RhiResult<()> {
        if self.compute.is_some() || self.render.is_some() {
            return self.invalid("Metal raster scope overlaps an active scope");
        };
        end_blit(&mut self.blit);
        let pass = render_pass_descriptor(begin)?;
        pass.setVisibilityResultBuffer(Some(&self.visibility_scratch));
        self.raster_extent = Some(raster_scope_extent(begin)?);
        self.render = Some(
            self.command_buffer
                .renderCommandEncoderWithDescriptor(&pass)
                .ok_or_else(|| {
                    RhiError::new(
                        RhiErrorKind::BackendFailure,
                        "Metal failed to create render encoder",
                    )
                })?,
        );
        Ok(())
    }
    fn raster_end(&mut self) -> RhiResult<()> {
        if self.render.is_none() {
            return self.invalid("Metal raster scope end without begin");
        };
        self.end_render();
        self.flush_visibility()
    }
    fn raster_begin_query(&mut self, set: &QuerySet, index: u32) -> RhiResult<()> {
        let render = self.render.as_deref().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "occlusion query outside raster scope",
            )
        })?;
        if self.active_query.is_some() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "nested Metal occlusion query",
            ));
        }
        let native = super::metal_query_set(set)?;
        if index >= native.count {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "occlusion query index exceeds query set",
            ));
        }
        if self.visibility_next >= 65_536 {
            return Err(RhiError::new(
                RhiErrorKind::OutOfMemory,
                "Metal visibility scratch is exhausted",
            ));
        }
        let offset = self.visibility_next * 8;
        self.visibility_next += 1;
        render.setVisibilityResultMode_offset(MTLVisibilityResultMode::Counting, offset);
        self.active_query = Some((set.clone(), index, offset));
        Ok(())
    }
    fn raster_end_query(&mut self, set: &QuerySet, index: u32) -> RhiResult<()> {
        let render = self.render.as_deref().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "occlusion query outside raster scope",
            )
        })?;
        let (active_set, active_index, offset) = self.active_query.take().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "occlusion query end without begin",
            )
        })?;
        if active_set.id() != set.id() || active_index != index {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "occlusion query end does not match begin",
            ));
        }
        render.setVisibilityResultMode_offset(MTLVisibilityResultMode::Disabled, 0);
        self.pending_queries.push((active_set, index, offset));
        Ok(())
    }
    fn raster_push_debug_group(&mut self, label: &str) -> RhiResult<()> {
        self.debug_push(label);
        Ok(())
    }
    fn raster_pop_debug_group(&mut self) -> RhiResult<()> {
        self.debug_pop();
        Ok(())
    }
    fn raster_insert_debug_marker(&mut self, label: &str) -> RhiResult<()> {
        self.debug_marker(label);
        Ok(())
    }
    fn raster_set_pipeline(&mut self, p: &RasterPipeline) -> RhiResult<()> {
        self.raster.pipeline = Some(p.clone());
        self.raster.immediates.clear();
        Ok(())
    }
    fn raster_set_bind_group(
        &mut self,
        index: BindGroupIndex,
        group: &BindGroup,
        offsets: &[u32],
    ) -> RhiResult<()> {
        let v = BoundGroup {
            index,
            group: group.clone(),
            dynamic_offsets: offsets.to_vec(),
        };
        Self::bound(&mut self.raster.groups, v, |x| x.index == index);
        Ok(())
    }
    fn raster_set_vertex_buffer(&mut self, slot: u32, b: &BufferBinding) -> RhiResult<()> {
        let v = (slot, b.clone());
        Self::bound(&mut self.raster.vertex_buffers, v, |x| x.0 == slot);
        Ok(())
    }
    fn raster_set_index_buffer(&mut self, b: &BufferBinding, f: IndexFormat) -> RhiResult<()> {
        self.raster.index = Some(BoundIndexBuffer {
            binding: b.clone(),
            format: f,
        });
        Ok(())
    }
    fn raster_set_viewport(&mut self, v: Viewport) -> RhiResult<()> {
        self.raster.viewport = Some(v);
        Ok(())
    }
    fn raster_set_scissor(&mut self, r: Rect) -> RhiResult<()> {
        self.raster.scissor = Some(r);
        Ok(())
    }
    fn raster_set_blend_constant(&mut self, c: Color) -> RhiResult<()> {
        self.raster.blend = c;
        Ok(())
    }
    fn raster_set_stencil_reference(&mut self, v: u32) -> RhiResult<()> {
        self.raster.stencil = v;
        Ok(())
    }
    fn raster_set_immediates(&mut self, w: &ImmediateWrite) -> RhiResult<()> {
        let v = w.clone();
        Self::bound(&mut self.raster.immediates, v, |x| x.offset == w.offset);
        Ok(())
    }
    fn raster_draw(
        &mut self,
        v: core::ops::Range<u32>,
        i: core::ops::Range<u32>,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        self.draw(v, i, 0, false)
    }
    fn raster_draw_indexed(
        &mut self,
        v: core::ops::Range<u32>,
        base: i32,
        i: core::ops::Range<u32>,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        self.draw(v, i, base, true)
    }
    fn raster_draw_indirect(
        &mut self,
        arguments: &Buffer,
        arguments_offset: u64,
        draw_count: u32,
        stride: u32,
        count: Option<(&Buffer, u64, u32)>,
        indexed: bool,
        _: &[ResourceUse],
    ) -> RhiResult<()> {
        if count.is_some() {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "Metal indirect-count draws are not implemented",
            )
            .at("MetalNativeEncoder::raster_draw_indirect"));
        }
        let encoder = self.render.as_deref().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "indirect draw outside raster scope",
            )
        })?;
        let pipeline = self.raster.pipeline.as_ref().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "indirect draw requires pipeline",
            )
        })?;
        let native = pipeline
            .native()
            .as_any()
            .downcast_ref::<super::super::pipeline::MetalRasterPipeline>()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::WrongDevice,
                    "raster pipeline is not backed by Metal",
                )
            })?;
        if indexed != self.raster.index.is_some() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "indirect indexed state does not match index binding",
            ));
        }
        let extent = self.raster_extent.ok_or_else(|| {
            RhiError::new(RhiErrorKind::InvalidUsage, "raster scope has no extent")
        })?;
        encoder.setRenderPipelineState(native.state());
        encoder.setDepthStencilState(native.depth_stencil());
        encoder.setStencilReferenceValue(self.raster.stencil);
        let primitive = &pipeline.descriptor().primitive;
        encoder.setCullMode(metal_cull_mode(primitive.cull_mode));
        encoder.setFrontFacingWinding(metal_winding(primitive.front_face));
        encoder.setTriangleFillMode(metal_fill_mode(primitive.polygon_mode)?);
        encoder.setBlendColorRed_green_blue_alpha(
            self.raster.blend.r,
            self.raster.blend.g,
            self.raster.blend.b,
            self.raster.blend.a,
        );
        encoder.setViewport(metal_viewport(self.raster.viewport.unwrap_or(
            Viewport::new(0.0, 0.0, extent.0 as f32, extent.1 as f32, 0.0, 1.0),
        )));
        encoder.setScissorRect(metal_scissor(
            self.raster
                .scissor
                .unwrap_or(Rect::new(0, 0, extent.0, extent.1)),
        ));
        bind_vertex_buffers(encoder, &self.raster.vertex_buffers)?;
        bind_raster_groups(encoder, native.binding_abi(), &self.raster.groups)?;
        bind_raster_immediates(encoder, native.binding_abi(), &self.raster.immediates)?;
        let arguments = metal_buffer(arguments)?;
        for n in 0..draw_count {
            let offset = arguments_offset
                .checked_add(u64::from(n).checked_mul(u64::from(stride)).ok_or_else(|| {
                    RhiError::new(RhiErrorKind::InvalidUsage, "indirect offset overflow")
                })?)
                .ok_or_else(|| {
                    RhiError::new(RhiErrorKind::InvalidUsage, "indirect offset overflow")
                })? as usize;
            unsafe {
                if let Some(index) = &self.raster.index {
                    let index_buffer = metal_buffer(&index.binding.buffer)?;
                    encoder.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(metal_primitive(primitive.topology),metal_index_type(index.format),&index_buffer.raw,index.binding.range.offset as usize,&arguments.raw,offset);
                } else {
                    encoder.drawPrimitives_indirectBuffer_indirectBufferOffset(
                        metal_primitive(primitive.topology),
                        &arguments.raw,
                        offset,
                    );
                }
            }
        }
        Ok(())
    }
    fn finish(mut self: Box<Self>) -> RhiResult<Box<dyn CommandBufferBackend>> {
        if self.compute.is_some() || self.render.is_some() {
            return self.invalid("Metal command encoder finished with an open GPU scope");
        };
        end_blit(&mut self.blit);
        Ok(Box::new(MetalNativeCommandBuffer {
            finished: Mutex::new(Some(FinishedMetalCommandBuffer {
                command_buffer: self.command_buffer.clone(),
                readbacks: std::mem::take(&mut self.readbacks),
                retained_staging: {
                    // Visibility results are copied by a blit encoder after
                    // raster end, therefore the scratch buffer must survive
                    // through the queue completion just like upload staging.
                    self.retained_buffers.push(self.visibility_scratch.clone());
                    std::mem::take(&mut self.retained_buffers)
                },
            })),
        }))
    }
}

impl MetalNativeEncoder {
    fn copy_buffer_texture(&mut self, copy: &BufferTextureCopy, to_texture: bool) -> RhiResult<()> {
        self.end_compute();
        self.end_render();
        let e = ensure_blit(&self.command_buffer, &mut self.blit)?;
        let stride = u64::from(copy.bytes_per_row)
            .checked_mul(u64::from(copy.rows_per_image))
            .ok_or_else(|| {
                RhiError::new(RhiErrorKind::InvalidUsage, "Metal image stride overflow")
            })?;
        if to_texture {
            let s = metal_buffer(&copy.buffer)?;
            let d = metal_texture(&copy.texture)?;
            for layer in 0..copy.texture_subresource.layer_count {
                unsafe {
                    e.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(&s.raw,(copy.buffer_offset+stride*u64::from(layer))as usize,copy.bytes_per_row as usize,stride as usize,size(copy.extent),&d.raw,(copy.texture_subresource.base_layer+layer)as usize,copy.texture_subresource.mip_level as usize,origin(copy.texture_origin));
                }
            }
        } else {
            let s = metal_texture(&copy.texture)?;
            let d = metal_buffer(&copy.buffer)?;
            for layer in 0..copy.texture_subresource.layer_count {
                unsafe {
                    e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(&s.raw,(copy.texture_subresource.base_layer+layer)as usize,copy.texture_subresource.mip_level as usize,origin(copy.texture_origin),size(copy.extent),&d.raw,(copy.buffer_offset+stride*u64::from(layer))as usize,copy.bytes_per_row as usize,stride as usize);
                }
            }
        }
        Ok(())
    }
}
impl CommandBufferBackend for MetalNativeCommandBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl MetalNativeCommandBuffer {
    pub(crate) fn is_available(&self) -> bool {
        self.finished
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_some()
    }
    pub(super) fn take(&self) -> Option<FinishedMetalCommandBuffer> {
        self.finished
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
    }
}
