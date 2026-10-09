//! Direct3D 12 lowering for a recorded raster scope.
//!
//! The portable recorder has already validated attachment compatibility and
//! draw bounds. This module owns only native state: attachment descriptors,
//! resource-to-render transitions, IA state, and draw calls.
//!
//! TODO(perf): Raster lowering currently creates one CPU-only RTV/DSV heap per
//! attachment use and retains those heaps in `CommittedBatch` until the batch
//! fence completes. This is a correct baseline: the CPU descriptor handles stay
//! valid for command-list execution and cannot be recycled while the list may
//! still reference them. Replace it with persistent RTV/DSV allocators only with
//! fence-keyed descriptor retirement, and cache/reuse views only when resource,
//! subresource range, format, and read-only flags match exactly. Attachment
//! ownership and `CompletionPoint` already provide the required RHI semantics;
//! no native heap or descriptor handle belongs in the public API.
//!
//! TODO(perf): Draw lowering currently rebinds the root signature, PSO, both
//! descriptor heaps and every root table for every draw. Add a command-list-local
//! state cache keyed by those native identities and invalidate it on list reset;
//! only changed slots need native calls. This is backend encoder state, not a
//! reason to add pipeline/binding state to the public recorder.

use std::collections::HashMap;

use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY_LINELIST, D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
    D3D_PRIMITIVE_TOPOLOGY_POINTLIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_UINT};

use crate::api::command::attachment::{
    ColorAttachmentView, DepthAttachmentMode, StencilAttachmentMode,
};
use crate::api::command::geometry::{ColorClearValue, LoadOp};
use crate::api::command::record::{RasterBegin, SecondaryRasterWork};
use crate::api::command::{AccessMask, IndexFormat, ResourceUse, TextureUseIntent};
use crate::api::command::{Color, RasterAttachmentClear, Rect, Viewport};
use crate::api::identity::ObjectId;
use crate::api::pipeline::{PrimitiveTopology, RasterPipeline};
use crate::api::presentation::FrameAttachment;
use crate::api::resource::buffer::{Buffer, BufferBinding};
use crate::api::resource::texture::{Extent3d, Texture, TextureDimension};
use crate::api::resource::view::{TextureView, TextureViewDimension};
use crate::backend::dx12::binding::Dx12BindGroup;
use crate::backend::dx12::binding::vocabulary::RegisterClass;
use crate::backend::dx12::failure::{Dx12Failure, ref_native};
use crate::backend::dx12::pipeline::Dx12RasterPipeline;
use crate::backend::dx12::platform::facts::dxgi_format;
use crate::backend::dx12::presentation::Dx12FrameAttachment;

use super::transfer::CommittedBatch;
use super::transition::Transitions;
use super::{dx12_buffer, dx12_texture};

pub(super) struct RasterScopeState {
    colors: Vec<Dx12RenderAttachment>,
    /// Portable color views remain available while the scope is open so an
    /// ordered clear can create an RTV over exactly the requested layer range.
    clear_colors: Vec<(u32, ColorAttachmentView, Option<u32>)>,
    /// CPU RTV heaps backing the null descriptors that fill sparse-MRT holes.
    /// A null RTV has no resource, so these are the only thing keeping the
    /// descriptor alive through the scope; dropped with the scope.  Never read —
    /// it is an ownership field: the handles it owns are referenced by the
    /// `color_handles` span handed to `OMSetRenderTargets`, and releasing the
    /// heap while that span is live would free the descriptors mid-draw.
    #[allow(dead_code)]
    null_rtv_heaps: Vec<ID3D12DescriptorHeap>,
    /// Multisample→single-sample color resolves to issue at scope end, one per
    /// attachment that declared `resolve`. Each carries the source (MSAA) and
    /// destination (single-sample) native resources and the destination view's
    /// subresource index. Resolved lazily in `lower_raster_end` so the source is
    /// still `RESOLVE_SOURCE` when the resolve fires.
    resolves: Vec<Dx12ColorResolve>,
    /// Parallel to `colors`: whether that color attachment is the source of a
    /// resolve. A resolve source leaves in `RESOLVE_SOURCE` (not `RENDER_TARGET`)
    /// once the resolve block has transitioned it, so `lower_raster_end` picks
    /// the leaving start state from this rather than assuming every color is a
    /// plain render target.
    color_resolves: Vec<bool>,
    depth_view: Option<TextureView>,
    depth_handle: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    depth_heap: Option<ID3D12DescriptorHeap>,
    depth_read_only: bool,
    depth_mode: Option<DepthAttachmentMode>,
    stencil_mode: Option<StencilAttachmentMode>,
    attachment_extent: Extent3d,
}

/// Minimal inherited state used while a D3D12 bundle records draw commands.
/// Secondary recordings must set viewport and scissor explicitly; attachments
/// and their dimensions are inherited from the parent direct list.
pub(super) fn secondary_scope_stub() -> RasterScopeState {
    RasterScopeState {
        colors: Vec::new(),
        clear_colors: Vec::new(),
        null_rtv_heaps: Vec::new(),
        resolves: Vec::new(),
        color_resolves: Vec::new(),
        depth_view: None,
        depth_handle: None,
        depth_heap: None,
        depth_read_only: false,
        depth_mode: None,
        stencil_mode: None,
        attachment_extent: Extent3d::d2(0, 0),
    }
}

/// One color-attachment resolve: source MSAA texture into a single-sample target.
struct Dx12ColorResolve {
    src: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    dst: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    dst_subresource: u32,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    dst_enter_state: D3D12_RESOURCE_STATES,
    dst_leave_state: D3D12_RESOURCE_STATES,
    frame: Option<FrameAttachment>,
}

/// The DX12-native shape shared by ordinary texture attachments and swapchain
/// frames. The portable distinction is consumed once, at construction; command
/// encoding below works only with a resource, its state round trip and a native
/// descriptor. Retention deliberately keeps the portable owner as well as the
/// COM resource because presentation ownership and texture-view lifetime remain
/// distinct contracts even though both lower to an `ID3D12Resource`.
struct Dx12RenderAttachment {
    resource: ID3D12Resource,
    handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    heap: ID3D12DescriptorHeap,
    enter_state: D3D12_RESOURCE_STATES,
    leave_state: D3D12_RESOURCE_STATES,
    retention: AttachmentRetention,
}

enum AttachmentRetention {
    Texture(TextureView),
    Frame(FrameAttachment),
}

/// A frame is usable by a raster draw only through the scope that attached it.
/// The scope owns its PRESENT/render-target transitions and keeps the acquired
/// attachment alive through submission; a draw's `ResourceUse::Frame` merely
/// repeats that same access for portable hazard tracking.
fn scope_contains_frame(
    scope: &RasterScopeState,
    frame: crate::api::presentation::AcquiredFrameId,
) -> bool {
    scope.colors.iter().any(|attachment| {
        matches!(
            &attachment.retention,
            AttachmentRetention::Frame(attached) if attached.frame_id() == frame
        )
    }) || scope.resolves.iter().any(|resolve| {
        resolve
            .frame
            .as_ref()
            .is_some_and(|attached| attached.frame_id() == frame)
    })
}

pub(super) fn lower_raster_begin(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    begin: &RasterBegin,
    _committed: &mut CommittedBatch,
) -> Result<RasterScopeState, Dx12Failure> {
    let mut entering = Transitions::default();
    let mut color_attachments = Vec::with_capacity(begin.colors.len());
    let clear_colors = begin
        .colors
        .iter()
        .map(|(location, color)| (*location, color.view.clone(), color.depth_slice))
        .collect();
    let mut resolves = Vec::new();
    let mut color_resolves = Vec::with_capacity(begin.colors.len());
    for (_, color) in &begin.colors {
        color_resolves.push(color.resolve.is_some());
        let attachment = lower_color_attachment(device, &color.view, color.depth_slice)?;
        // A resolve target is part of the attachment (section 31.1): it is a
        // single-sampled texture the multisampled source resolves into at scope
        // end. Resolve is only meaningful alongside a multisampled source.
        if let Some(target) = &color.resolve {
            let src = attachment.resource.clone();
            let (dst, format, dst_enter_state, dst_leave_state, frame) = match target {
                ColorAttachmentView::Texture(target_view) => (
                    dx12_texture(target_view.texture())?.resource().clone(),
                    dxgi_format(target_view.format()).ok_or_else(|| {
                        unsupported(
                            "a resolve target whose format has no DXGI mapping",
                            "DX12 cannot resolve into it",
                        )
                    })?,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                ),
                ColorAttachmentView::Frame(frame) => {
                    let native = frame
                        .native()
                        .as_any()
                        .downcast_ref::<Dx12FrameAttachment>()
                        .ok_or_else(|| {
                            unsupported(
                                "a resolve frame this device did not acquire",
                                "its native drawable belongs to another backend",
                            )
                        })?;
                    (
                        native.resource().clone(),
                        dxgi_format(frame.format()).ok_or_else(|| {
                            unsupported(
                                "a presentation resolve format without DXGI mapping",
                                "DX12 cannot resolve into it",
                            )
                        })?,
                        D3D12_RESOURCE_STATE_PRESENT,
                        D3D12_RESOURCE_STATE_PRESENT,
                        Some(frame.clone()),
                    )
                }
                _ => {
                    return Err(unsupported(
                        "a resolve attachment introduced after this DX12 backend",
                        "the backend has no verified resolve lowering for it",
                    ));
                }
            };
            // Whole-subresource resolve for the common single-layer, base-mip
            // single-sample target. A layered/resolved-mip target would need a
            // subresource index derived from the view.
            resolves.push(Dx12ColorResolve {
                src,
                dst,
                dst_subresource: 0,
                format,
                dst_enter_state,
                dst_leave_state,
                frame,
            });
        }
        entering.push(
            &attachment.resource,
            attachment.enter_state,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        color_attachments.push(attachment);
    }

    // Sparse portable MRT is a legal scope: `begin.colors` carries each
    // attachment's own `location`, so locations need not be dense. D3D12
    // `OMSetRenderTargets` takes a dense array indexed by attachment slot, so a
    // hole (a location with no attachment) must be filled with an explicit null
    // RTV descriptor. The null descriptor's format is taken from the first real
    // attachment; writes to a null slot are discarded by the fixed-function
    // stage, which is exactly what "no attachment at this slot" means.
    let max_location = begin
        .colors
        .iter()
        .map(|(location, _)| *location)
        .max()
        .unwrap_or(0);
    let null_format = begin
        .colors
        .first()
        .and_then(|(_, color)| dxgi_format(color.view.format()));
    let mut handle_by_location: Vec<Option<D3D12_CPU_DESCRIPTOR_HANDLE>> =
        vec![None; (max_location + 1) as usize];
    for ((location, _), attachment) in begin.colors.iter().zip(&color_attachments) {
        handle_by_location[*location as usize] = Some(attachment.handle);
    }
    // One CPU RTV heap per hole, created eagerly. A null RTV has no backing
    // resource; it lives exactly as long as the scope, matching the real
    // attachments' per-descriptor heap lifetime.
    let mut null_rtv_heaps = Vec::new();
    if let Some(format) = null_format {
        for slot in handle_by_location.iter_mut() {
            if slot.is_none() {
                let heap = cpu_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV)?;
                let handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
                let desc = D3D12_RENDER_TARGET_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_RTV {
                            MipSlice: 0,
                            PlaneSlice: 0,
                        },
                    },
                };
                // `pResource = None` creates an RTV with no backing resource;
                // writes to this slot are discarded rather than reaching any
                // memory. This is the D3D12-native spelling of a sparse MRT hole.
                unsafe { device.CreateRenderTargetView(None, Some(&desc), handle) };
                *slot = Some(handle);
                null_rtv_heaps.push(heap);
            }
        }
    }
    let color_handles = handle_by_location
        .iter()
        .map(|handle| handle.expect("every MRT slot is either a real or a null RTV descriptor"))
        .collect::<Vec<_>>();
    let (depth_stencil, depth_view, depth_heap, depth_read_only) =
        if let Some(depth) = &begin.depth_stencil {
            let native = dx12_texture(depth.view.texture())?;
            let read_only = matches!(depth.depth, Some(DepthAttachmentMode::ReadOnly) | None)
                && matches!(depth.stencil, Some(StencilAttachmentMode::ReadOnly) | None);
            entering.push(
                native.resource(),
                D3D12_RESOURCE_STATE_COMMON,
                if read_only {
                    D3D12_RESOURCE_STATE_DEPTH_READ
                } else {
                    D3D12_RESOURCE_STATE_DEPTH_WRITE
                },
            );
            let (heap, handle) = create_dsv(device, &depth.view, depth.depth, depth.stencil)?;
            (
                Some(handle),
                Some(depth.view.clone()),
                Some(heap),
                read_only,
            )
        } else {
            (None, None, None, false)
        };
    entering.record(list);
    unsafe {
        list.OMSetRenderTargets(
            color_handles.len() as u32,
            Some(color_handles.as_ptr()),
            false,
            depth_stencil.as_ref().map(std::ptr::from_ref),
        );
    }
    // Clear each real attachment at its own location. `color_handles` is indexed
    // by attachment slot (sparse, with null RTVs at holes), so a real attachment
    // may sit at a higher index than its position in `begin.colors`; the zip
    // below must therefore look each attachment up by stored location rather
    // than by enumeration order.
    for (location, attachment) in &begin.colors {
        let handle = color_handles[*location as usize];
        if let LoadOp::Clear(value) = attachment.load {
            unsafe { list.ClearRenderTargetView(handle, &clear_color(value), None) };
        }
    }
    if let (Some(depth), Some(handle)) = (&begin.depth_stencil, depth_stencil) {
        let mut flags = D3D12_CLEAR_FLAGS(0);
        let mut clear_depth = 1.0;
        let mut clear_stencil = 0u8;
        if let Some(DepthAttachmentMode::ReadWrite {
            load: LoadOp::Clear(value),
            ..
        }) = depth.depth
        {
            flags |= D3D12_CLEAR_FLAG_DEPTH;
            clear_depth = value;
        }
        if let Some(StencilAttachmentMode::ReadWrite {
            load: LoadOp::Clear(value),
            ..
        }) = depth.stencil
        {
            flags |= D3D12_CLEAR_FLAG_STENCIL;
            clear_stencil = value as u8;
        }
        if flags != D3D12_CLEAR_FLAGS(0) {
            unsafe { list.ClearDepthStencilView(handle, flags, clear_depth, clear_stencil, None) };
        }
    }
    let attachment_extent = begin
        .colors
        .first()
        .map(|(_, color)| color.view.extent())
        .or_else(|| {
            begin
                .depth_stencil
                .as_ref()
                .map(|depth| depth.view.extent())
        })
        .ok_or_else(|| {
            unsupported(
                "a raster scope without attachments",
                "portable validation must reject it before DX12 lowering",
            )
        })?;
    Ok(RasterScopeState {
        colors: color_attachments,
        clear_colors,
        null_rtv_heaps,
        resolves,
        color_resolves,
        depth_view,
        depth_handle: depth_stencil,
        depth_heap,
        depth_read_only,
        depth_mode: begin.depth_stencil.as_ref().and_then(|depth| depth.depth),
        stencil_mode: begin.depth_stencil.as_ref().and_then(|depth| depth.stencil),
        attachment_extent,
    })
}

fn lower_color_attachment(
    device: &ID3D12Device,
    view: &ColorAttachmentView,
    depth_slice: Option<u32>,
) -> Result<Dx12RenderAttachment, Dx12Failure> {
    match view {
        ColorAttachmentView::Texture(view) => {
            let resource = dx12_texture(view.texture())?.resource().clone();
            let (heap, handle) = create_rtv(device, view, depth_slice)?;
            Ok(Dx12RenderAttachment {
                resource,
                handle,
                heap,
                enter_state: D3D12_RESOURCE_STATE_COMMON,
                leave_state: D3D12_RESOURCE_STATE_COMMON,
                retention: AttachmentRetention::Texture(view.clone()),
            })
        }
        ColorAttachmentView::Frame(frame) => {
            if depth_slice.is_some() {
                return Err(unsupported(
                    "a depth slice on a presentation attachment",
                    "a frame attachment is not a 3D texture",
                ));
            }
            let native = frame
                .native()
                .as_any()
                .downcast_ref::<Dx12FrameAttachment>()
                .ok_or_else(|| {
                    unsupported(
                        "a frame attachment this device did not acquire",
                        "its native drawable belongs to another backend",
                    )
                })?;
            let resource = native.resource().clone();
            let (heap, handle) = create_frame_rtv(device, &resource, frame.format())?;
            Ok(Dx12RenderAttachment {
                resource,
                handle,
                heap,
                // PRESENT is numerically COMMON, but keeping the symbolic state
                // here protects the swapchain ownership invariant from being
                // erased by the native unification.
                enter_state: D3D12_RESOURCE_STATE_PRESENT,
                leave_state: D3D12_RESOURCE_STATE_PRESENT,
                retention: AttachmentRetention::Frame(frame.clone()),
            })
        }
        _ => Err(unsupported(
            "a color attachment introduced after this DX12 backend",
            "the backend has no verified render-target lowering for it",
        )),
    }
}

fn create_frame_rtv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    format: crate::api::format::TextureFormat,
) -> Result<(ID3D12DescriptorHeap, D3D12_CPU_DESCRIPTOR_HANDLE), Dx12Failure> {
    let format = dxgi_format(format).ok_or_else(|| {
        unsupported(
            "a presentation format without DXGI mapping",
            "DX12 cannot create its RTV",
        )
    })?;
    let heap = cpu_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV)?;
    let handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    let desc = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_RTV {
                MipSlice: 0,
                PlaneSlice: 0,
            },
        },
    };
    unsafe { device.CreateRenderTargetView(resource, Some(&desc), handle) };
    Ok((heap, handle))
}

/// Clears selected attachments without ending their raster scope.
///
/// The native clear commands operate on RTV/DSV descriptors directly, so this
/// deliberately does not rebind `OMSetRenderTargets`: the scope's attachments
/// remain current for the draw that follows.  That preserves the recorded
/// `query end → clear → draw` order used by the occlusion example.
pub(super) fn lower_raster_clear(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    clear: &RasterAttachmentClear,
    scope: &RasterScopeState,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let rect = clear_rect(clear)?;
    for (location, value) in &clear.colors {
        let index = scope
            .clear_colors
            .iter()
            .position(|(candidate, _, _)| candidate == location)
            .ok_or_else(|| {
                unsupported(
                    "a raster clear color location absent from the DX12 scope",
                    "portable validation must reject unattached locations before lowering",
                )
            })?;
        let (_, view, depth_slice) = &scope.clear_colors[index];
        let (handle, extra_heap) = match view {
            ColorAttachmentView::Texture(view) => {
                let (heap, handle) = create_rtv_range(
                    device,
                    view,
                    *depth_slice,
                    clear.base_layer,
                    clear.layer_count,
                )?;
                (handle, Some(heap))
            }
            ColorAttachmentView::Frame(_) => {
                if clear.base_layer != 0 || clear.layer_count != 1 {
                    return Err(unsupported(
                        "a layered clear of a presentation attachment",
                        "an acquired DX12 frame has exactly one layer",
                    ));
                }
                (scope.colors[index].handle, None)
            }
            _ => {
                return Err(unsupported(
                    "a color attachment introduced after this DX12 backend",
                    "the DX12 raster clear lowerer has no native descriptor for it",
                ));
            }
        };
        unsafe { list.ClearRenderTargetView(handle, &clear_color(*value), Some(&[rect])) };
        if let Some(heap) = extra_heap {
            committed.raster_descriptor_heaps.push(heap);
        }
    }

    if clear.depth.is_some() || clear.stencil.is_some() {
        let view = scope.depth_view.as_ref().ok_or_else(|| {
            unsupported(
                "a depth/stencil clear without a DX12 depth attachment",
                "portable validation must reject it before lowering",
            )
        })?;
        let (handle, extra_heap) = if clear.base_layer == 0 && clear.layer_count == 1 {
            (
                scope.depth_handle.ok_or_else(|| {
                    unsupported(
                        "a DX12 depth attachment without a DSV",
                        "raster-scope construction must create its DSV",
                    )
                })?,
                None,
            )
        } else {
            let (heap, handle) = create_dsv_range(
                device,
                view,
                scope.depth_mode,
                scope.stencil_mode,
                clear.base_layer,
                clear.layer_count,
            )?;
            (handle, Some(heap))
        };
        let mut flags = D3D12_CLEAR_FLAGS(0);
        if clear.depth.is_some() {
            flags |= D3D12_CLEAR_FLAG_DEPTH;
        }
        if clear.stencil.is_some() {
            flags |= D3D12_CLEAR_FLAG_STENCIL;
        }
        unsafe {
            list.ClearDepthStencilView(
                handle,
                flags,
                clear.depth.unwrap_or(1.0),
                clear.stencil.unwrap_or(0) as u8,
                Some(&[rect]),
            )
        };
        if let Some(heap) = extra_heap {
            committed.raster_descriptor_heaps.push(heap);
        }
    }
    Ok(())
}

fn clear_rect(
    clear: &RasterAttachmentClear,
) -> Result<windows::Win32::Foundation::RECT, Dx12Failure> {
    let right = clear.rect.right().ok_or_else(|| {
        unsupported(
            "a raster clear rectangle whose right edge overflows",
            "portable validation must reject an overflowing rectangle",
        )
    })?;
    let bottom = clear.rect.bottom().ok_or_else(|| {
        unsupported(
            "a raster clear rectangle whose bottom edge overflows",
            "portable validation must reject an overflowing rectangle",
        )
    })?;
    Ok(windows::Win32::Foundation::RECT {
        left: i32::try_from(clear.rect.x).map_err(|_| {
            unsupported(
                "a raster clear rectangle wider than i32",
                "D3D12 RECT uses LONG",
            )
        })?,
        top: i32::try_from(clear.rect.y).map_err(|_| {
            unsupported(
                "a raster clear rectangle taller than i32",
                "D3D12 RECT uses LONG",
            )
        })?,
        right: i32::try_from(right).map_err(|_| {
            unsupported(
                "a raster clear rectangle wider than i32",
                "D3D12 RECT uses LONG",
            )
        })?,
        bottom: i32::try_from(bottom).map_err(|_| {
            unsupported(
                "a raster clear rectangle taller than i32",
                "D3D12 RECT uses LONG",
            )
        })?,
    })
}

pub(super) fn lower_raster_draw_view(
    list: &ID3D12GraphicsCommandList,
    draw: &RasterDrawView<'_>,
    uses: &[ResourceUse],
    scope: &RasterScopeState,
    committed: &mut CommittedBatch,
    indexed: bool,
) -> Result<(), Dx12Failure> {
    if indexed != draw.index.is_some() {
        return Err(unsupported(
            "typed indexed draw state",
            "an indexed draw requires an index buffer and a non-indexed draw must not use one",
        ));
    }
    lower_raster_draw_inner(list, draw, uses, scope, committed, true, true, false)
}

/// Borrowed current graphics state plus the one draw's scalar arguments.
/// Native encoders pass this directly without constructing a draw packet.
pub(super) struct RasterDrawView<'a> {
    pub(super) pipeline: &'a RasterPipeline,
    pub(super) groups: &'a [crate::api::command::record::BoundGroup],
    pub(super) vertex_buffers: &'a [(u32, BufferBinding)],
    pub(super) index: Option<&'a crate::api::command::record::BoundIndexBuffer>,
    pub(super) viewport: Option<Viewport>,
    pub(super) scissor: Option<Rect>,
    pub(super) blend_constant: Color,
    pub(super) stencil_reference: u32,
    pub(super) range: core::ops::Range<u32>,
    pub(super) instances: core::ops::Range<u32>,
    pub(super) base_vertex: i32,
    pub(super) immediates: &'a [crate::api::command::record::ImmediateWrite],
}
/// Writes a draw's state and geometry.  A D3D12 bundle cannot contain resource
/// barriers, so inherited raster work uses this same encoder with transitions
/// disabled and emits them on the parent direct list around `ExecuteBundle`.
pub(super) fn lower_raster_draw_inner(
    list: &ID3D12GraphicsCommandList,
    draw: &RasterDrawView<'_>,
    uses: &[ResourceUse],
    scope: &RasterScopeState,
    committed: &mut CommittedBatch,
    record_transitions: bool,
    set_descriptor_heaps: bool,
    inherit_dynamic_state: bool,
) -> Result<(), Dx12Failure> {
    let pipeline = draw
        .pipeline
        .native()
        .as_any()
        .downcast_ref::<Dx12RasterPipeline>()
        .ok_or_else(|| {
            unsupported(
                "a raster pipeline this device did not create",
                "its native state belongs to another backend",
            )
        })?;
    let mut groups = Vec::with_capacity(draw.groups.len());
    for bound in draw.groups {
        let group = bound
            .group
            .native()
            .as_any()
            .downcast_ref::<Dx12BindGroup>()
            .ok_or_else(|| {
                unsupported(
                    "a bind group this device did not create",
                    "its descriptor table belongs to another backend",
                )
            })?;
        groups.push((bound, group));
    }
    let mut entering = Transitions::default();
    let mut leaving = Transitions::default();
    let mut buffers = HashMap::<ObjectId, (Buffer, D3D12_RESOURCE_STATES)>::new();
    let mut textures = HashMap::<ObjectId, (Texture, D3D12_RESOURCE_STATES)>::new();
    for resource_use in uses {
        match resource_use {
            ResourceUse::Buffer(use_) => {
                let state = buffer_state(use_.access);
                buffers
                    .entry(use_.buffer.id())
                    .and_modify(|(_, prior)| *prior |= state)
                    .or_insert_with(|| (use_.buffer.clone(), state));
            }
            ResourceUse::Texture(use_) => match use_.intent {
                TextureUseIntent::ColorAttachment
                | TextureUseIntent::DepthStencilRead
                | TextureUseIntent::DepthStencilWrite => {}
                TextureUseIntent::ShaderRead | TextureUseIntent::ShaderReadWrite => {
                    let state = texture_state(use_.access);
                    textures
                        .entry(use_.texture.id())
                        .and_modify(|(_, prior)| *prior |= state)
                        .or_insert_with(|| (use_.texture.clone(), state));
                }
                _ => {
                    return Err(unsupported(
                        "a raster draw texture use outside shader bindings",
                        "copy and resolve uses have separate lowerings",
                    ));
                }
            },
            ResourceUse::Frame(frame_use) => {
                if !scope_contains_frame(scope, frame_use.frame) {
                    return Err(unsupported(
                        "a presentation frame used by a raster draw outside its current scope",
                        "DX12 only lowers a frame as the scope attachment that owns its PRESENT-to-render-target transition",
                    ));
                }
                // `lower_raster_begin` already transitioned and retained this
                // exact attachment, and `lower_raster_end` restores PRESENT.
                // Draw-use metadata repeats that attachment access for hazard
                // tracking; it must not emit a second transition here.
            }
            ResourceUse::AccelerationStructure(_) => {
                return Err(unsupported(
                    "an acceleration structure used by a raster draw",
                    "DX12 acceleration-structure binding lowering is not enabled",
                ));
            }
            ResourceUse::Query(_) => {}
            _ => {
                return Err(unsupported(
                    "a resource use introduced after this DX12 backend",
                    "the backend has no verified raster transition for it",
                ));
            }
        }
    }
    for (buffer, state) in buffers.values() {
        let native = dx12_buffer(buffer)?;
        entering.push(native.resource(), D3D12_RESOURCE_STATE_COMMON, *state);
        leaving.push(native.resource(), *state, D3D12_RESOURCE_STATE_COMMON);
        committed.raster_buffers.push(buffer.clone());
    }
    for (texture, state) in textures.values() {
        let native = dx12_texture(texture)?;
        entering.push(native.resource(), D3D12_RESOURCE_STATE_COMMON, *state);
        leaving.push(native.resource(), *state, D3D12_RESOURCE_STATE_COMMON);
        committed.raster_textures.push(texture.clone());
    }
    let mut vertex_views = Vec::with_capacity(draw.vertex_buffers.len());
    for (slot, binding) in draw.vertex_buffers {
        let buffer = dx12_buffer(&binding.buffer)?;
        let stride = draw
            .pipeline
            .descriptor()
            .vertex_input
            .buffers
            .get(*slot as usize)
            .ok_or_else(|| {
                unsupported(
                    "a vertex buffer slot absent from the pipeline",
                    "portable validation should have refused it",
                )
            })?
            .stride;
        vertex_views.push((
            *slot,
            D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: unsafe { buffer.resource().GetGPUVirtualAddress() }
                    + binding.range.offset,
                SizeInBytes: u32::try_from(binding.range.size).map_err(|_| {
                    unsupported(
                        "a vertex buffer range larger than 4 GiB",
                        "D3D12 IA views use u32 sizes",
                    )
                })?,
                StrideInBytes: u32::try_from(stride).map_err(|_| {
                    unsupported(
                        "a vertex stride larger than u32",
                        "D3D12 IA views use u32 strides",
                    )
                })?,
            },
        ));
    }
    if record_transitions {
        entering.record(list);
    }
    unsafe {
        list.SetGraphicsRootSignature(pipeline.root_signature());
        list.SetPipelineState(pipeline.pipeline_state());
        for write in draw.immediates {
            let (parameter, destination) = pipeline
                .immediate_root_parameter(write.offset, write.bytes.len() as u32)
                .ok_or_else(|| {
                    unsupported(
                        "an immediate write outside the DX12 root-constant layout",
                        "portable validation must keep writes within declared ranges",
                    )
                })?;
            let values: Vec<u32> = write
                .bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("4-byte immediate word")))
                .collect();
            list.SetGraphicsRoot32BitConstants(
                parameter,
                values.len() as u32,
                values.as_ptr().cast(),
                destination,
            );
        }
        if set_descriptor_heaps {
            if let Some((_, group)) = groups.first() {
                // D3D12 permits precisely one CBV/SRV/UAV and one sampler heap to
                // be active. A group's table addresses are offsets in these shared
                // device heaps, so bind both before setting graphics roots.
                list.SetDescriptorHeaps(&[
                    Some(group.view_heap().clone()),
                    Some(group.sampler_heap().clone()),
                ]);
            }
        }
        for (bound, group) in &groups {
            if let Some(parameter) = pipeline.view_root_parameter(bound.index.get()) {
                list.SetGraphicsRootDescriptorTable(parameter, group.view_table());
            }
            if let Some(parameter) = pipeline.sampler_root_parameter(bound.index.get()) {
                list.SetGraphicsRootDescriptorTable(parameter, group.sampler_table());
            }
            bind_graphics_dynamic_root_descriptors(
                list,
                pipeline.dynamic_root_parameters(bound.index.get()),
                group,
                &bound.dynamic_offsets,
            )?;
        }
        list.IASetPrimitiveTopology(primitive_topology(
            draw.pipeline.descriptor().primitive.topology,
        ));
        // Bind each slot independently: the portable raster scope may bind slot
        // 3 while slots 0..2 are intentionally absent, and an IA call starting
        // at zero would silently reinterpret that view as slot zero.
        for (slot, view) in &vertex_views {
            list.IASetVertexBuffers(*slot, Some(std::slice::from_ref(view)));
        }
        let viewport = match draw.viewport {
            Some(viewport) => viewport,
            None if inherit_dynamic_state => {
                // The parent direct list established attachment-sized defaults.
                // A bundle must leave them untouched to inherit that state.
                Viewport::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            }
            None => default_viewport(scope),
        };
        if !inherit_dynamic_state || draw.viewport.is_some() {
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: viewport.x,
                TopLeftY: viewport.y,
                Width: viewport.width,
                Height: viewport.height,
                MinDepth: viewport.min_depth,
                MaxDepth: viewport.max_depth,
            }]);
        }
        let scissor = match draw.scissor {
            Some(scissor) => scissor,
            None if inherit_dynamic_state => Rect::new(0, 0, 0, 0),
            None => default_scissor(scope),
        };
        if !inherit_dynamic_state || draw.scissor.is_some() {
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: scissor.x as i32,
                top: scissor.y as i32,
                right: scissor.right().unwrap_or(u32::MAX) as i32,
                bottom: scissor.bottom().unwrap_or(u32::MAX) as i32,
            }]);
        }
        list.OMSetBlendFactor(Some(&[
            draw.blend_constant.r,
            draw.blend_constant.g,
            draw.blend_constant.b,
            draw.blend_constant.a,
        ]));
        list.OMSetStencilRef(draw.stencil_reference);
        if let Some(index) = &draw.index {
            let buffer = dx12_buffer(&index.binding.buffer)?;
            let format = match index.format {
                IndexFormat::Uint16 => DXGI_FORMAT_R16_UINT,
                IndexFormat::Uint32 => DXGI_FORMAT_R32_UINT,
            };
            list.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: buffer.resource().GetGPUVirtualAddress()
                    + index.binding.range.offset,
                SizeInBytes: u32::try_from(index.binding.range.size).map_err(|_| {
                    unsupported(
                        "an index buffer range larger than 4 GiB",
                        "D3D12 IA views use u32 sizes",
                    )
                })?,
                Format: format,
            }));
            list.DrawIndexedInstanced(
                draw.range.end - draw.range.start,
                draw.instances.end - draw.instances.start,
                draw.range.start,
                draw.base_vertex,
                draw.instances.start,
            );
        } else {
            list.DrawInstanced(
                draw.range.end - draw.range.start,
                draw.instances.end - draw.instances.start,
                draw.range.start,
                draw.instances.start,
            );
        }
    }
    if record_transitions {
        leaving.record(list);
    }
    committed.raster_pipelines.push(draw.pipeline.clone());
    committed
        .bind_groups
        .extend(draw.groups.iter().map(|group| group.group.clone()));
    Ok(())
}

fn bind_graphics_dynamic_root_descriptors(
    list: &ID3D12GraphicsCommandList,
    parameters: Option<&[u32]>,
    group: &Dx12BindGroup,
    offsets: &[u32],
) -> Result<(), Dx12Failure> {
    let parameters = parameters.unwrap_or_default();
    let buffers = group.dynamic_buffers();
    if parameters.len() != buffers.len() || buffers.len() != offsets.len() {
        return Err(unsupported(
            "a raster dynamic binding layout that disagrees with its bind group",
            "portable validation establishes one dynamic offset per buffer element",
        ));
    }
    for ((parameter, buffer), offset) in parameters.iter().zip(buffers).zip(offsets) {
        let address = buffer.address(*offset)?;
        unsafe {
            match buffer.class() {
                RegisterClass::ConstantBuffer => {
                    list.SetGraphicsRootConstantBufferView(*parameter, address)
                }
                RegisterClass::ShaderResource => {
                    list.SetGraphicsRootShaderResourceView(*parameter, address)
                }
                RegisterClass::UnorderedAccess => {
                    list.SetGraphicsRootUnorderedAccessView(*parameter, address)
                }
                RegisterClass::Sampler => unreachable!("dynamic offsets are buffer bindings"),
            }
        }
    }
    Ok(())
}

/// Lowers a finished inherited raster packet into D3D12 bundles.  Resource
/// barriers and descriptor-heap selection belong to the parent direct list:
/// D3D12 forbids barriers in a bundle and bundles inherit the parent's heaps.
pub(super) fn lower_secondary_raster_work(
    device: &ID3D12Device,
    parent: &ID3D12GraphicsCommandList,
    work: &SecondaryRasterWork,
    scope: &RasterScopeState,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    if let Some(native) = work
        .native()
        .as_any()
        .downcast_ref::<super::native::Dx12NativeCommandBuffer>()
    {
        let mut bundle = native.take().ok_or_else(|| {
            unsupported(
                "a secondary raster command buffer submitted more than once",
                "a D3D12 bundle allocator remains owned by its first accepted submission",
            )
        })?;
        if !bundle.secondary {
            return Err(unsupported(
                "a primary command buffer used as secondary raster work",
                "ExecuteBundle accepts only D3D12 bundle command lists",
            ));
        }
        let leaving = secondary_draw_transitions(parent, work.uses(), scope, committed)?;
        if let Some((views, samplers)) = bundle.secondary_heaps.take() {
            unsafe { parent.SetDescriptorHeaps(&[Some(views.clone()), Some(samplers.clone())]) };
            committed.raster_descriptor_heaps.push(views);
            committed.raster_descriptor_heaps.push(samplers);
        }
        // A bundle inherits dynamic state. Establish the parent attachment
        // defaults first; a bundle draw with explicit state overwrites these.
        set_inherited_dynamic_state(parent, scope);
        unsafe { parent.ExecuteBundle(&bundle.list) };
        leaving.record(parent);
        committed
            .secondary_bundles
            .push((bundle.allocator, bundle.list));
        committed.absorb(bundle.committed);
        return Ok(());
    }

    Err(unsupported(
        "a secondary raster command buffer without DX12 native backing",
        "direct DX12 encoding requires the child bundle to be closed before parent execution",
    ))
}

fn set_inherited_dynamic_state(parent: &ID3D12GraphicsCommandList, scope: &RasterScopeState) {
    let viewport = default_viewport(scope);
    let scissor = default_scissor(scope);
    unsafe {
        parent.RSSetViewports(&[D3D12_VIEWPORT {
            TopLeftX: viewport.x,
            TopLeftY: viewport.y,
            Width: viewport.width,
            Height: viewport.height,
            MinDepth: viewport.min_depth,
            MaxDepth: viewport.max_depth,
        }]);
        parent.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
            left: scissor.x as i32,
            top: scissor.y as i32,
            right: scissor.right().unwrap_or(u32::MAX) as i32,
            bottom: scissor.bottom().unwrap_or(u32::MAX) as i32,
        }]);
    }
}

/// Emits the parent-side state transitions for one bundle and returns the
/// matching restoration barriers.  Keep this intentionally parallel to the
/// direct draw encoder: a bundle inherits resource state but cannot carry a
/// `ResourceBarrier` command of its own.
fn secondary_draw_transitions(
    parent: &ID3D12GraphicsCommandList,
    uses: &[ResourceUse],
    scope: &RasterScopeState,
    committed: &mut CommittedBatch,
) -> Result<Transitions, Dx12Failure> {
    let mut entering = Transitions::default();
    let mut leaving = Transitions::default();
    let mut buffers = HashMap::<ObjectId, (Buffer, D3D12_RESOURCE_STATES)>::new();
    let mut textures = HashMap::<ObjectId, (Texture, D3D12_RESOURCE_STATES)>::new();
    for resource_use in uses {
        match resource_use {
            ResourceUse::Buffer(use_) => {
                let state = buffer_state(use_.access);
                buffers
                    .entry(use_.buffer.id())
                    .and_modify(|(_, prior)| *prior |= state)
                    .or_insert_with(|| (use_.buffer.clone(), state));
            }
            ResourceUse::Texture(use_) => match use_.intent {
                TextureUseIntent::ColorAttachment
                | TextureUseIntent::DepthStencilRead
                | TextureUseIntent::DepthStencilWrite => {}
                TextureUseIntent::ShaderRead | TextureUseIntent::ShaderReadWrite => {
                    let state = texture_state(use_.access);
                    textures
                        .entry(use_.texture.id())
                        .and_modify(|(_, prior)| *prior |= state)
                        .or_insert_with(|| (use_.texture.clone(), state));
                }
                _ => {
                    return Err(unsupported(
                        "a secondary raster texture use outside shader bindings",
                        "copy and resolve uses have separate lowerings",
                    ));
                }
            },
            ResourceUse::Frame(frame_use) => {
                if !scope_contains_frame(scope, frame_use.frame) {
                    return Err(unsupported(
                        "a presentation frame used by secondary raster work outside its parent scope",
                        "DX12 bundles inherit only the parent scope attachment transitions",
                    ));
                }
            }
            ResourceUse::AccelerationStructure(_) => {
                return Err(unsupported(
                    "an acceleration structure used by secondary raster work",
                    "DX12 acceleration-structure binding lowering is not enabled",
                ));
            }
            ResourceUse::Query(_) => {}
            _ => {
                return Err(unsupported(
                    "a resource use introduced after this DX12 backend",
                    "the backend has no verified raster transition for it",
                ));
            }
        }
    }
    for (buffer, state) in buffers.values() {
        let native = dx12_buffer(buffer)?;
        entering.push(native.resource(), D3D12_RESOURCE_STATE_COMMON, *state);
        leaving.push(native.resource(), *state, D3D12_RESOURCE_STATE_COMMON);
        committed.raster_buffers.push(buffer.clone());
    }
    for (texture, state) in textures.values() {
        let native = dx12_texture(texture)?;
        entering.push(native.resource(), D3D12_RESOURCE_STATE_COMMON, *state);
        leaving.push(native.resource(), *state, D3D12_RESOURCE_STATE_COMMON);
        committed.raster_textures.push(texture.clone());
    }
    entering.record(parent);
    Ok(leaving)
}

/// Encodes indirect draw metadata from the native encoder's borrowed graphics
/// state without constructing or retaining a recorded raster packet.
pub(super) fn lower_raster_indirect_view(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    setup: &RasterDrawView<'_>,
    argument_buffer: &Buffer,
    arguments_offset: u64,
    draw_count: u32,
    stride: u32,
    count_meta: Option<(&Buffer, u64, u32)>,
    uses: &[ResourceUse],
    scope: &RasterScopeState,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    lower_raster_draw_inner(list, &setup, uses, scope, committed, true, true, false)?;
    let indexed = setup.index.is_some();
    let kind = if indexed {
        D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED
    } else {
        D3D12_INDIRECT_ARGUMENT_TYPE_DRAW
    };
    let desc = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: kind,
        Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0::default(),
    };
    let signature_desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &desc,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    unsafe {
        device
            .CreateCommandSignature(
                &signature_desc,
                None::<&ID3D12RootSignature>,
                &mut signature,
            )
            .map_err(|error| {
                Dx12Failure::Native(crate::backend::dx12::ffi::NativeError::new(
                    &error,
                    "ID3D12Device::CreateCommandSignature",
                ))
            })?;
    }
    let signature = signature.ok_or_else(|| {
        unsupported(
            "a missing raster command signature",
            "CreateCommandSignature returned success without a signature",
        )
    })?;
    let arguments = dx12_buffer(argument_buffer)?;
    let count = match count_meta {
        Some((buffer, offset, max)) => Some((dx12_buffer(buffer)?, offset, max)),
        None => None,
    };
    let mut entering = Transitions::default();
    entering.push(
        arguments.resource(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    );
    if let Some((buffer, _, _)) = &count {
        entering.push(
            buffer.resource(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        );
    }
    entering.record(list);
    unsafe {
        list.ExecuteIndirect(
            &signature,
            count.as_ref().map_or(draw_count, |(_, _, max)| *max),
            arguments.resource(),
            arguments_offset,
            count.as_ref().map(|(buffer, _, _)| buffer.resource()),
            count.as_ref().map_or(0, |(_, offset, _)| *offset),
        );
    }
    let mut leaving = Transitions::default();
    leaving.push(
        arguments.resource(),
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_COMMON,
    );
    if let Some((buffer, _, _)) = &count {
        leaving.push(
            buffer.resource(),
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        );
    }
    leaving.record(list);
    committed.indirect_buffers.push(argument_buffer.clone());
    if let Some((buffer, _, _)) = count_meta {
        committed.indirect_buffers.push(buffer.clone());
    }
    committed.command_signatures.push(signature);
    Ok(())
}

pub(super) fn lower_raster_end(
    list: &ID3D12GraphicsCommandList,
    scope: RasterScopeState,
    committed: &mut CommittedBatch,
) {
    let RasterScopeState {
        colors,
        null_rtv_heaps: _,
        resolves,
        color_resolves,
        depth_view,
        depth_heap,
        depth_read_only,
        attachment_extent: _,
        ..
    } = scope;

    // Multisample color resolve, before the source leaves RENDER_TARGET. Each
    // attachment's resolve target receives the resolved single-sample result.
    if !resolves.is_empty() {
        let mut resolve_transitions = Transitions::default();
        for op in &resolves {
            resolve_transitions.push(
                &op.src,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
            );
            resolve_transitions.push(
                &op.dst,
                op.dst_enter_state,
                D3D12_RESOURCE_STATE_RESOLVE_DEST,
            );
        }
        resolve_transitions.record(list);
        for op in &resolves {
            // SAFETY: both resources are live DXGI allocations owned by the
            // attached texture views, and the resolve is issued while the source
            // is still `RESOLVE_SOURCE`.
            unsafe { list.ResolveSubresource(&op.dst, op.dst_subresource, &op.src, 0, op.format) };
        }
    }

    let mut leaving = Transitions::default();
    // A resolve source leaves from `RESOLVE_SOURCE` (it was transitioned there
    // by the resolve block above), not from `RENDER_TARGET`. Transitioning a
    // resolve source from `RENDER_TARGET` to COMMON would name a start state it
    // is no longer in, which D3D12 treats as a resource-state violation.
    for (index, attachment) in colors.iter().enumerate() {
        let from = if color_resolves.get(index).copied().unwrap_or(false) {
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE
        } else {
            D3D12_RESOURCE_STATE_RENDER_TARGET
        };
        leaving.push(&attachment.resource, from, attachment.leave_state);
    }
    for op in &resolves {
        // The resolved destination returns to COMMON with the source's leaving
        // phase, so a later copy/read is legal.
        leaving.push(
            &op.dst,
            D3D12_RESOURCE_STATE_RESOLVE_DEST,
            op.dst_leave_state,
        );
    }
    if let Some(view) = &depth_view {
        if let Ok(texture) = dx12_texture(view.texture()) {
            leaving.push(
                texture.resource(),
                if depth_read_only {
                    D3D12_RESOURCE_STATE_DEPTH_READ
                } else {
                    D3D12_RESOURCE_STATE_DEPTH_WRITE
                },
                D3D12_RESOURCE_STATE_COMMON,
            );
        }
    }
    leaving.record(list);
    for attachment in colors {
        committed.raster_descriptor_heaps.push(attachment.heap);
        match attachment.retention {
            AttachmentRetention::Texture(view) => committed.raster_views.push(view),
            AttachmentRetention::Frame(frame) => committed.raster_frames.push(frame),
        }
    }
    for op in resolves {
        if let Some(frame) = op.frame {
            committed.raster_frames.push(frame);
        }
    }
    if let Some(view) = depth_view {
        committed.raster_views.push(view);
    }
    if let Some(heap) = depth_heap {
        committed.raster_descriptor_heaps.push(heap);
    }
}

fn create_rtv(
    device: &ID3D12Device,
    view: &TextureView,
    depth_slice: Option<u32>,
) -> Result<(ID3D12DescriptorHeap, D3D12_CPU_DESCRIPTOR_HANDLE), Dx12Failure> {
    create_rtv_range(device, view, depth_slice, 0, 1)
}

/// Creates an RTV selecting a consecutive range relative to a texture view.
/// Scope begin uses the first layer only; `RasterAttachmentClear` is allowed to
/// name any validated subset and therefore needs this explicit range form.
fn create_rtv_range(
    device: &ID3D12Device,
    view: &TextureView,
    depth_slice: Option<u32>,
    relative_base_layer: u32,
    layer_count: u32,
) -> Result<(ID3D12DescriptorHeap, D3D12_CPU_DESCRIPTOR_HANDLE), Dx12Failure> {
    let texture = dx12_texture(view.texture())?;
    let descriptor = view.descriptor();
    let multisampled = view.texture().descriptor().sample_count > 1;
    let format = dxgi_format(view.format()).ok_or_else(|| {
        unsupported(
            "a color attachment format without DXGI mapping",
            "DX12 cannot create its RTV",
        )
    })?;
    let heap = cpu_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV)?;
    let handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    let base_layer = descriptor
        .base_layer
        .checked_add(relative_base_layer)
        .ok_or_else(|| {
            unsupported(
                "a raster clear layer index overflow",
                "validated layer ranges must fit the native RTV",
            )
        })?;
    // A portable D2 view selects one array layer. D3D12 requires the ARRAY
    // descriptor form whenever the resource has multiple layers; TEXTURE2D
    // would silently target layer zero and ignore `base_layer`.
    let desc = if matches!(view.texture().descriptor().dimension, TextureDimension::D3)
        && matches!(descriptor.dimension, TextureViewDimension::D3)
    {
        if relative_base_layer != 0 || layer_count != 1 {
            return Err(unsupported(
                "a layered clear of a 3D color attachment",
                "the DX12 3D RTV path selects one explicit W slice",
            ));
        }
        let slice = depth_slice.ok_or_else(|| {
            unsupported(
                "a 3D color attachment without a depth slice",
                "the portable attachment requires an explicit 3D slice",
            )
        })?;
        D3D12_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_RTV {
                    MipSlice: descriptor.base_mip,
                    FirstWSlice: slice,
                    WSize: 1,
                },
            },
        }
    } else if multisampled
        && matches!(view.texture().descriptor().dimension, TextureDimension::D2)
        && (view.texture().descriptor().array_layers > 1
            || matches!(descriptor.dimension, TextureViewDimension::D2Array))
    {
        D3D12_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2DMSArray: D3D12_TEX2DMS_ARRAY_RTV {
                    FirstArraySlice: base_layer,
                    ArraySize: layer_count,
                },
            },
        }
    } else if matches!(view.texture().descriptor().dimension, TextureDimension::D2)
        && matches!(
            descriptor.dimension,
            TextureViewDimension::D2 | TextureViewDimension::D2Array
        )
        && (view.texture().descriptor().array_layers > 1
            || matches!(descriptor.dimension, TextureViewDimension::D2Array))
    {
        D3D12_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_RTV {
                    MipSlice: descriptor.base_mip,
                    FirstArraySlice: base_layer,
                    ArraySize: layer_count,
                    PlaneSlice: 0,
                },
            },
        }
    } else if multisampled
        && matches!(view.texture().descriptor().dimension, TextureDimension::D2)
        && matches!(descriptor.dimension, TextureViewDimension::D2)
    {
        if depth_slice.is_some() || relative_base_layer != 0 || layer_count != 1 {
            return Err(unsupported(
                "a layered clear of a non-array 2D color attachment",
                "only a 2D-array RTV can select several attachment layers",
            ));
        }
        D3D12_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2DMS,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2DMS: D3D12_TEX2DMS_RTV::default(),
            },
        }
    } else if matches!(view.texture().descriptor().dimension, TextureDimension::D2)
        && matches!(descriptor.dimension, TextureViewDimension::D2)
    {
        if depth_slice.is_some() || relative_base_layer != 0 || layer_count != 1 {
            return Err(unsupported(
                "a layered clear of a non-array 2D color attachment",
                "only a 2D-array RTV can select several attachment layers",
            ));
        }
        D3D12_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_RTV {
                    MipSlice: descriptor.base_mip,
                    PlaneSlice: 0,
                },
            },
        }
    } else {
        return Err(unsupported(
            "a color attachment view dimension",
            "DX12 supports this backend's 2D and explicit-slice 3D RTV paths only",
        ));
    };
    unsafe { device.CreateRenderTargetView(texture.resource(), Some(&desc), handle) };
    Ok((heap, handle))
}

fn create_dsv(
    device: &ID3D12Device,
    view: &TextureView,
    depth: Option<DepthAttachmentMode>,
    stencil: Option<StencilAttachmentMode>,
) -> Result<(ID3D12DescriptorHeap, D3D12_CPU_DESCRIPTOR_HANDLE), Dx12Failure> {
    create_dsv_range(device, view, depth, stencil, 0, 1)
}

fn create_dsv_range(
    device: &ID3D12Device,
    view: &TextureView,
    depth: Option<DepthAttachmentMode>,
    stencil: Option<StencilAttachmentMode>,
    relative_base_layer: u32,
    layer_count: u32,
) -> Result<(ID3D12DescriptorHeap, D3D12_CPU_DESCRIPTOR_HANDLE), Dx12Failure> {
    let texture = dx12_texture(view.texture())?;
    let descriptor = view.descriptor();
    if !matches!(view.texture().descriptor().dimension, TextureDimension::D2)
        || !matches!(
            descriptor.dimension,
            TextureViewDimension::D2 | TextureViewDimension::D2Array
        )
    {
        // A multisampled 2D depth attachment is legitimate MSAA; the DSV form is
        // the same 2D descriptor regardless of sample count (D3D12 reads the
        // sample count from the resource). Only 2D (and 2D-array) depth is
        // lowered here.
        return Err(unsupported(
            "a non-2D depth attachment view",
            "DX12 raster lowering currently implements 2D DSV descriptors",
        ));
    }
    let format = dxgi_format(view.format()).ok_or_else(|| {
        unsupported(
            "a depth attachment format without DXGI mapping",
            "DX12 cannot create its DSV",
        )
    })?;
    let mut flags = D3D12_DSV_FLAG_NONE;
    if matches!(depth, Some(DepthAttachmentMode::ReadOnly)) {
        flags |= D3D12_DSV_FLAG_READ_ONLY_DEPTH;
    }
    // A depth-only format has no stencil plane. Marking that absent plane
    // read-only produces an invalid DSV descriptor; CreateDepthStencilView is
    // void, so some drivers defer the error until a later GPU call.
    if matches!(stencil, Some(StencilAttachmentMode::ReadOnly)) {
        flags |= D3D12_DSV_FLAG_READ_ONLY_STENCIL;
    }
    let heap = cpu_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV)?;
    let handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    let base_layer = descriptor
        .base_layer
        .checked_add(relative_base_layer)
        .ok_or_else(|| {
            unsupported(
                "a depth/stencil clear layer index overflow",
                "validated layer ranges must fit the native DSV",
            )
        })?;
    let multisampled = view.texture().descriptor().sample_count > 1;
    let desc = if multisampled
        && (view.texture().descriptor().array_layers > 1
            || matches!(descriptor.dimension, TextureViewDimension::D2Array))
    {
        D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY,
            Flags: flags,
            Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2DMSArray: D3D12_TEX2DMS_ARRAY_DSV {
                    FirstArraySlice: base_layer,
                    ArraySize: layer_count,
                },
            },
        }
    } else if multisampled {
        if relative_base_layer != 0 || layer_count != 1 {
            return Err(unsupported(
                "a layered clear of a non-array 2D depth attachment",
                "only a 2D-array DSV can select several attachment layers",
            ));
        }
        D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DMS,
            Flags: flags,
            Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2DMS: D3D12_TEX2DMS_DSV::default(),
            },
        }
    } else if view.texture().descriptor().array_layers > 1
        || matches!(descriptor.dimension, TextureViewDimension::D2Array)
    {
        D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
            Flags: flags,
            Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                    MipSlice: descriptor.base_mip,
                    FirstArraySlice: base_layer,
                    ArraySize: layer_count,
                },
            },
        }
    } else {
        if relative_base_layer != 0 || layer_count != 1 {
            return Err(unsupported(
                "a layered clear of a non-array 2D depth attachment",
                "only a 2D-array DSV can select several attachment layers",
            ));
        }
        D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            Flags: flags,
            Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_DSV {
                    MipSlice: descriptor.base_mip,
                },
            },
        }
    };
    unsafe { device.CreateDepthStencilView(texture.resource(), Some(&desc), handle) };
    Ok((heap, handle))
}

fn cpu_heap(
    device: &ID3D12Device,
    ty: D3D12_DESCRIPTOR_HEAP_TYPE,
) -> Result<ID3D12DescriptorHeap, Dx12Failure> {
    unsafe {
        device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: ty,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        })
    }
    .map_err(|error| ref_native(&error))
}

fn default_viewport(scope: &RasterScopeState) -> crate::api::command::Viewport {
    let extent = scope.attachment_extent;
    crate::api::command::Viewport::new(
        0.0,
        0.0,
        extent.width as f32,
        extent.height as f32,
        0.0,
        1.0,
    )
}
fn default_scissor(scope: &RasterScopeState) -> crate::api::command::Rect {
    let extent = scope.attachment_extent;
    crate::api::command::Rect::new(0, 0, extent.width, extent.height)
}
fn clear_color(value: ColorClearValue) -> [f32; 4] {
    match value {
        ColorClearValue::Float(v) => v,
        ColorClearValue::Sint(v) => v.map(|x| x as f32),
        ColorClearValue::Uint(v) => v.map(|x| x as f32),
        _ => [0.0; 4],
    }
}
fn primitive_topology(
    topology: PrimitiveTopology,
) -> windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY {
    match topology {
        PrimitiveTopology::PointList => D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
        PrimitiveTopology::LineList => D3D_PRIMITIVE_TOPOLOGY_LINELIST,
        PrimitiveTopology::LineStrip => D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
        PrimitiveTopology::TriangleList => D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        PrimitiveTopology::TriangleStrip => D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
        _ => D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
    }
}
fn unsupported(what: &'static str, why: &'static str) -> Dx12Failure {
    Dx12Failure::Unsupported { what, why }
}

fn buffer_state(access: AccessMask) -> D3D12_RESOURCE_STATES {
    if access.contains(AccessMask::SHADER_WRITE) {
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    } else {
        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
            | D3D12_RESOURCE_STATE_INDEX_BUFFER
            | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    }
}

fn texture_state(access: AccessMask) -> D3D12_RESOURCE_STATES {
    if access.contains(AccessMask::SHADER_WRITE) {
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    } else {
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A presentation attachment has no `TextureView`. Keep the default dynamic
    /// state source separate from texture-view retention so frame-only scopes do
    /// not regress into the historical `.first().expect()` panic.
    #[test]
    fn default_dynamic_state_uses_the_recorded_attachment_extent() {
        let scope = RasterScopeState {
            colors: Vec::new(),
            clear_colors: Vec::new(),
            null_rtv_heaps: Vec::new(),
            resolves: Vec::new(),
            color_resolves: Vec::new(),
            depth_view: None,
            depth_handle: None,
            depth_heap: None,
            depth_read_only: false,
            depth_mode: None,
            stencil_mode: None,
            attachment_extent: Extent3d::d2(640, 480),
        };

        let viewport = default_viewport(&scope);
        assert_eq!((viewport.width, viewport.height), (640.0, 480.0));
        let scissor = default_scissor(&scope);
        assert_eq!((scissor.width, scissor.height), (640, 480));
    }
}
