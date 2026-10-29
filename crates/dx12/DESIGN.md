# Direct3D 12 Backend Design

`crates/dx12` is the Windows-native lowering of the Core execution contract to
D3D12, DXGI, and DXR. It owns native resources and command objects while
preserving Core operation meanings and making support dependent on actual
device facts.

## Ownership and architecture

`Dx12Instance` discovers DXGI adapters and constructs a concrete `Device<Dx12>`.
Device state owns the native device, direct and optional compute queues,
fences, descriptor heaps, command allocators, staging/upload state, pipeline
objects, and backend-private allocation pools. Narrow modules divide resource
creation, descriptors, pipelines, encoding/submission, memory, sparse
resources, mesh, and ray operations.

The encoder lowers the caller's ordered steps and explicit resource
transitions to native command lists. It does not maintain a global hazard graph
or infer resource states. A queue gate enforces native queue synchronization;
it is not a scheduler or a resource planner.

## Submission and lifetime

Submission retains command allocators, uploads, referenced resources, and other
native dependencies until the relevant fence establishes completion. An error
after command-list execution or a failed signal can leave accepted work
uncertain; it is not reported as a clean cancellation. Device removal, fence
completion, and observer cancellation remain distinct lifecycle facts.

Graphics and optional compute work use native queues and fences. Explicit
cross-queue waits represent caller-provided dependencies. The backend does not
insert frame policy or renderer resource retirement.

## Resources, shaders, and native limits

Default buffer and texture allocations use backend-owned heap pages when native
requirements permit. A resource retains its allocation lease until the native
resource is destroyed. Dedicated fallback, explicit placement, sparse binding,
external imports, and caller-selected aliasing keep their separate contracts;
the default pool does not make alias or hazard decisions.

Shader formats are explicit routes. WGSL translation uses the maintained
Naga/DXC path; Direct-DXIL is a separate native artifact path. Descriptor
layouts and shader register mappings must agree before pipeline creation. A
success in one artifact route does not admit a different shader route.

Optional capabilities such as ray tracing, ray query, mesh shaders, sparse
resources, descriptor arrays, dynamic raster state, texture atomics, and
GPU-counted draws are reported only when the selected device and the exact
lowering route satisfy their requirements. DXR state objects are not covered
by the PSO cache; a ray pipeline cache route remains unavailable.

## Verified descriptor-array boundary

Fixed, fully bound, compute WGSL uniform-buffer and storage-buffer
non-uniform-indexing routes are implemented and verified. The WGSL array route
does not imply support for runtime or partially bound arrays, dynamic offsets,
non-compute stages, or layouts whose shader and binding metadata do not match.
These shapes must continue to fail explicitly. The historical hand-authored
Direct-DXIL UBO failure was not recovered; it remains an unknown historical
artifact and does not invalidate the independently verified WGSL route.

## Public feature coverage

The public route includes graphics and compute, multiple render targets,
indirect drawing and dispatch, queries, secondary recording, standalone clears
and resolves, placed memory, native/external interop, and pipeline caches.
Format, shader-model, resource descriptor, and device-tier checks still apply.

Raster supports up to eight color attachments, per-target blend and write
masks, depth/stencil, read-only depth/stencil views, and attachment clears and
resolves. Sample counts and formats remain device-dependent.

Dynamic uniform and bounded raw/structured storage offsets are validated and
retained through completion. Descriptor arrays and indexing are gated by
device capabilities and descriptor restrictions. Push constants have
stage-specific visibility. Secondary bundles preserve primary graphics state;
supported view-instancing tiers expose up to four views with view masks.

Acceleration structures, ray pipelines, ray queries, mesh shaders, sparse
resources, and graphics/compute queues require the corresponding reported
capability and supported descriptor. GPU-count multi-draw and source wave-size
selection are also device-gated.

For compressed textures, DX12 exposes BC1–BC7 through device-checked creation,
block-aligned upload/copy, and sampling. Individual formats, dimensions, and
usages are checked through `texture_support`. ETC2/EAC and ASTC have no DXGI
format mapping and are unsupported by this route.

## Non-goals

DX12 does not implement RenderGraph scheduling, global resource-state tracking,
CPU command replay, generic device-loss notification policy, arbitrary shader
format conversion, or a DXR state-object cache. See [`TODO.md`](../../TODO.md) for
the open cross-backend work ledger.
