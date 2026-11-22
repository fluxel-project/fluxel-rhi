# Direct3D 12 Backend Design

`fluxel-rhi-dx12` is the Windows-native implementation of the shared RHI
contract. It translates typed Core operations into D3D12 command lists, DXGI
presentation, and, where admitted, DXR operations. The application still owns
render-graph planning, resource-use ordering, frame pacing, and retirement
policy. This crate owns the native objects that execute that already planned
work and reports facts about its completion.

The shared contract is designed first for RenderGraph and rendering-pipeline
implementations. This backend keeps execution thin and low-cost by preserving
direct D3D12 capabilities and exposing their execution facts to graph planning;
application-level developer convenience is secondary.

The public entry points are `Dx12Instance`, `Dx12`, and the typed
`Device<Dx12>`. Native D3D12 and DXGI types remain private to this crate.
Callers work through the Core API, so a DX12 object cannot be confused with an
object from a different backend or device.

## What makes this backend different

D3D12 is a deferred, explicit API. An encoder creates a D3D12 command list;
recording validates the Core operation, emits D3D12 calls, and gathers every
native object that must outlive execution. Submission executes that list on a
native queue and returns a fence-backed completion point. Therefore command
recording is separate from execution, and completion is an observed GPU fact,
not an inference from a successful Rust call.

The backend supports a direct graphics queue and opens a separate compute
queue when the requested device exposes that lane. Queue ordering and waits
are explicit parts of the Core submission plan. The backend serializes native
queue access and fence allocation, but it does not reorder work, create
dependencies, or turn the queues into a render scheduler.

DXGI presentation is also explicit. A surface borrows a live Win32 `HWND`,
owns a flip-model swapchain generation and its RTVs, and issues `Present` only
when the acquired image lease is handed off. Host window lifetime and window
policy remain outside the crate.

Before opening a device, `Instance::probe_surface` accepts a borrowed Win32
window handle and an adapter from that instance. DXGI can report whether
the adapter lists the current monitor among its outputs. This information
is advisory on hybrid systems. Device-bound surface configuration uses
the actual D3D12 queue and HWND to qualify the swapchain.

## Layers

The implementation is organized by responsibility rather than by a second
public abstraction layer:

```text
Core typed API and backend SPI
        |
Dx12Instance / Dx12Device
        |
  +-----+-------------------+---------------------+
  |                         |                     |
resource, memory,       pipelines, shaders,    surface and
descriptor modules      raster, compute, ray    DXGI image leases
  |                         |                     |
encoder records D3D12 command lists and retained native owners
        |
submission preflight, direct/compute queues, D3D12 fences
        |
completion polling or asynchronous fence wait
```

`backend/mod.rs` creates the device-wide state: `ID3D12Device`, queues,
fences, descriptor heaps, command allocator pools, upload pages, default
resource pools, pipeline helper caches, and the in-flight owner lists.
Focused modules own resource descriptions and views, descriptor layouts and
writes, pipeline construction, shader compilation, queue mechanics, surface
leases, and feature families such as sparse, mesh, or ray work. The encoder is
the only layer that turns Core command operations into native list calls.

The Core SPI preserves a concrete `Dx12` type for devices, buffers, textures,
views, bind groups, encoders, surfaces, and completion points. The public
`Device<Dx12>` identity is checked before a native object is used; the crate
does not expose COM pointers as general RHI resource handles.

## Device opening and capability facts

`Dx12Instance::new` opens a DXGI factory. Adapter enumeration happens through
the Core instance API, and `open` creates the D3D12 device only for the chosen
adapter. Opening records fixed device facts such as feature level, shader
model, resource-heap tier, enhanced-barrier support, mesh and ray support,
sparse support, descriptor-indexing prerequisites, and view-instancing tier.

Those facts become `DeviceCapabilities` and `DeviceLimits`. A feature is
reported only when both the native device and this backend's exact lowering
route can support it. For example, a D3D12 feature tier alone does not imply
that every shader format, descriptor shape, texture format, or stage is
accepted.

Power preferences use DXGI's preferred adapter order and resolve the result by
LUID into the ordinary instance-scoped ordinal. Software selection explicitly
requests WARP. DXGI model IDs and available driver version or memory properties
are diagnostic values. Device opening queries the native subgroup width range
once, and the required-wave-size pipeline path combines that range with its
shader model and shader route checks.

Opening checks named feature requirements and minimum capacities against the
same capability snapshot returned to the caller. DX12 checks these requirements
after native device probing and before creating queues, fences, or descriptor
heaps.

Descriptor-table limits combine D3D12 binding-tier ceilings and the
shader-visible heap capacities used by this backend. A layout reserves the
declared maximum for each range, including runtime-sized arrays. Root
descriptors for dynamic uniform buffers use the root signature budget.
Tables declare visibility to all stages, so pipeline layout admission checks
the combined ranges before serializing the root signature.

The GPU-counted indirect draw profile distinguishes a single native count
operation from WGSL vertex and instance index adjustment. DX12 copies small
batches and packs larger batches on the GPU. Both adjusted routes use
temporary arguments, retain the native count operation, and keep the raster
pass active. CPU recording grows per record for the bounded copy and per
batch for the compute conversion. `Device::multi_draw_count_plan` reports
the number of native counted draws before buffer allocation. Ordinary
and small-copy routes record one native batch. The pack route uses the
same stride-dependent batch limit as command recording.

The optional debug callback is a diagnostic aid. It does not make driver
validation output a substitute for completion or visual correctness.

## Resource and descriptor ownership

Buffers and textures use D3D12 resources with typed Core wrappers. Ordinary
default-heap allocations come from backend-owned pages when native alignment
and heap rules allow it; their allocation lease stays alive with the resource.
Committed resources and explicitly placed resources retain their independent
native backing. The default allocator neither aliases resources on behalf of a
caller nor decides when an aliasing barrier is safe.
After an aliasing handoff, the renderer orders initialization of the new
identity before reading it. Placed render-target and depth-stencil textures
also require native initialization. A full clear or full subresource copy
establishes contents. Native discard can establish the resource metadata
needed for later writes, with the texel contents left undefined.

Explicit default heaps support synchronous `MakeResident` and `Evict` on the
physical heap. A new heap starts with one residency reference. Placed resources
share that heap's residency. The caller supplies ordering across all GPU lanes
that access the heap. The residency methods touch the explicit heap only, so
they add no tracking to submission or to the backend-owned allocator.
Sparse allocations use the same calls on their backing heaps. Their reserved
textures and tile mappings are unchanged across residency changes. Completion
must cover accesses through every mapping as well as queued mapping operations.
Locally committed default buffers and textures own their implicit heaps.
Only those allocation routes mark their resources eligible for committed
residency. Placed, pool-backed, sparse and external resources require their
actual physical owner. The native resource itself is retained for an accepted
asynchronous residency request on a committed allocation.
Explicit committed creation bypasses pool selection and retains the usual
descriptor and allocation checks. Ordinary creation continues to pick its
existing allocation route.
Explicit upload and readback heaps share an access state with their placed
buffers. Map leases keep their physical byte ranges registered until native
`Unmap`. Residency changes require every map lease to end. Mapping requires
a held residency reference and completion of any accepted asynchronous
residency operation. Distinct placed buffers can map disjoint ranges at the
same time. Pool-backed and ordinary CPU-visible buffers use their existing
mapping path.
When the opened device exposes `ID3D12Device3`, asynchronous residency uses
`EnqueueMakeResident` with the caller's budget flag and a private operation
fence. The fence does not advance either command timeline. The backend retains
the physical heap until native completion, including when the caller drops the
receipt. A failed observer registration quarantines the owner without changing
the accepted enqueue or its fence.
An explicit residency wait reuses native command-list segmentation. The
target queue waits at the encoder boundary on the private residency fence.
The recorded work retains the fence through queue completion. The renderer
orders later sparse mapping steps and other lanes through the compiled plan.

The memory observation domain can report the number and largest size of
contiguous free ranges across retained default-pool pages. The observation
uses the pool lock and requires no GPU wait or native budget query. A request
fits a reported range only if it also meets the page's heap class and the
resource's placement alignment.

Descriptor layout, descriptor writing, and bind-group ownership are separate
from command recording. Device-owned shader-visible heaps provide GPU tables;
the CPU descriptor pool owns non-shader-visible views. Pipeline layout creation
maps the validated Core binding layout to D3D12 root parameters and descriptor
tables. Raster and compute encoders bind those tables only after checking the
pipeline, group, and dynamic-offset shapes agree.

Sampled cube and cube-array views preserve their native SRV dimensions for
color and depth/stencil formats. View admission checks cube support on the
typed SRV format. A sampled depth/stencil cube view selects one aspect.
A cube-array SRV can start at any in-bounds face and covers a whole number
of cubes. A cube view selects exactly six faces. At a nonzero starting face,
DX12 uses a cube-array SRV with one cube and preserves the public cube
dimension and shader binding type.

Upload and readback have dedicated paths. CPU writes are copied from upload
storage into default resources; texture readback targets a compatible readback
buffer and requires the declared copy state. These helpers insert only their
own required native transitions. They do not infer transitions or hazards for
ordinary caller-owned resources.

CPU texture writes repack caller rows into encoder-owned upload pages using
the native row pitch and placement alignment. The same pages serve buffer
writes, and submitted pages stay checked out until the covering queue fence
retires the encoder.
The direct texture staging writer receives the actual row and image strides
in initialized storage. A canceled write records no transitions or copy.
Depth and stencil uploads require the direct lane because the helper
restores the depth-write state.

Host mappings use upload buffers for writes and readback buffers for reads.
Clones share a single native mapping lease, including when a readback ticket
copies bytes into host-owned storage. Callers may explicitly wait on a chosen
completion before acquiring a lease and remain responsible for ordering later
GPU work. Reported UMA topology does not change the buffer heap contract.
Private staging readback publishes the first covering backend fence signal
from the copy's lane. A ticket owns its byte range and is consumed by `read_private` after
that signal completes. Discarded recordings and uncertain queue handoffs
produce failed tickets. Caller-supplied completion and write exclusion are
still required for direct mapping of shared readback buffers.
An explicit GPU upload placement uses a committed `GPU_UPLOAD` heap when
`OPTIONS16.GPUUploadHeapSupported` is true. It shares the host-write and
GPU-read contract of an upload buffer, including the fixed `GENERIC_READ`
state. The placement specifies the allocation route. The runtime controls
actual residency.
Explicit custom placement matches the requested cache policy and pool against
the node's equivalent upload, readback or GPU upload properties. Unified
memory uses L0 for both pool choices. Discrete device-local upload uses L1
under the GPU upload capability. Each custom allocation retains the access
and state contract of its selected profile.

`Stencil8` uses a D24S8 allocation and exposes its stencil plane. Textures
with sampled usage use `R24G8_TYPELESS` and an `X24_TYPELESS_G8_UINT` SRV.
Depth testing, writes, clears and transitions are unavailable on `Stencil8`.
The hidden depth plane of locally created textures stays in its initial
native state. Sparse and shared imports remain gated.

## Shader and pipeline routes

Shader route selection is explicit and precedes pipeline creation:

- Direct DXIL is accepted as a D3D12 artifact after entry-point validation.
- HLSL is a direct source route when the selected shader model and stage are
  admitted; DXC compilation occurs for the pipeline specialization that needs
  it.
- WGSL follows the maintained Naga-to-HLSL/DXC route. The compiler validates
  entries, bindings, stage declarations, workgroup size, overrides, and
  required feature declarations against the pipeline layout.

The result is a `Dx12Shader` plus entry-specific native bytecode used to build
raster, compute, mesh, or ray pipeline objects. Pipeline caching applies to
the compatible fixed pipeline forms it owns. DXR state objects are separate
and are intentionally outside that cache.

Direct DXIL, direct HLSL, and WGSL-lowered DXIL are different artifact routes.
Success on one says nothing about the others. The same applies to optional
shader capabilities: admission combines the device fact, required Core
feature, accepted stage/storage class, and translator support.

`create_sampled_texture_view` can map each shader-visible color component to
an ordinary view component or a constant zero or one. DX12 creates only a
shader-resource descriptor for this view, even when the texture itself also
allows storage or render-attachment usage. The support query checks the same
sampled-view format, dimension, and subresource constraints as creation.

`create_texture_view_with_usage` selects a nonempty subset of the source
texture's sampled, storage, and attachment uses. Support and creation share
admission. Descriptor ownership follows that subset. This permits a mip
chain for sampling alongside a single-mip storage or attachment view.
Combined depth-stencil textures use aspect-specific sampled views and an
all-aspect attachment view. When a default view inherits both usages, DX12
keeps the sampled role for a single aspect and the attachment role for the
combined aspect. If the requested aspects leave none of the texture's
sampled or attachment roles available, creation fails before allocating a
descriptor.

Default views require at least one sampled, storage, or attachment role.
A copy-only texture exposes no compatible view formats. Copies address
texture subresources directly.

A 3D color attachment slice selects one W coordinate in a single mip.
The RTV carries that coordinate while resource transitions and attachment
alias checks still address the full mip. Pass clears, including integer
load and rectangular clears, use the selected RTV. Render-pass MSAA resolves
use matching typed 2D views and write only the render-area rectangle.
Typeless allocations are admitted through compatible typed views.

Dual-source blending uses a single color target at slot zero. WGSL
`dual_source_blending` emits the two sources as `SV_Target0` and
`SV_Target1`. Pipeline admission checks the selected pixel program's DXIL
output signature for the channels used by blending. A raster support
query with dual-source blending compiles the selected fragment source
when needed to obtain that signature, without creating a PSO.

Storage texture allocation and view creation require a typed UAV and at
least one typed access. Bind-group layout support and creation require the
typed load bit for read, the typed store bit for write, and both bits for
read-write bindings. WGSL shader declarations are checked against that exact
layout access. Atomics additionally require operation and stage admission.

Explicit color buffer/texture copies share placement and pitch validation in
both directions. When `texel_aligned_color_copies` is reported, the offset
and row pitch may each be aligned to a single texel or compressed block.
Other DX12 devices use 512-byte offsets and 256-byte row pitches for those
copies. Depth and stencil use their dedicated copy paths.
`buffer_to_texture_support` and `texture_to_buffer_support` validate the
same resource usage, subresources, and footprint bounds as recording.
The queries neither record commands nor reserve GPU scratch storage.
Buffer-texture copies require a single-sample texture.

Bind-group creation checks sampler comparison mode and ordinary filtering
before reserving descriptor tables. Filtering layouts accept ordinary point
samplers. Nonfiltering layouts require ordinary point sampling for min,
mag, and mip. Comparison layouts admit comparison samplers with point or
linear filtering. These checks cover each populated array entry.

Native NV12 and P010 are separately gated by the device's 2D texture
support, native plane count, and both sampled view formats. They require
even width and height, one mip, and one sample. A view selects one plane and
its fixed UNORM format. NV12 uses `R8Unorm` and `Rg8Unorm`, while P010 uses
`R16Unorm` and `Rg16Unorm`. Plane 1 coordinates and extents are measured in
half-resolution chroma texels. Buffer copies preserve the native plane
storage words, including P010's 16-bit words. Transitions address the
selected native plane subresources. Callers supply synchronization and any
color conversion. Sparse textures and shared import have separate admission.

Shared NV12 and P010 imports also validate the producer's resource format,
dimensions, layout, and shader-resource flags. Each acquire or release
selects both planes of the requested layers and supplies one known state
pair for them. Acquire puts the producer fence wait before the transition.
Release puts the signal after it. Queue segments retain the fence and
recorded commands retain the imported native resource.

Holes in sampled and storage texture descriptor arrays use null DX12 views
with the binding's native view dimension and numeric type. Storage holes
use the declared UAV format. Shader code selects only populated elements.

DX12 storage buffer entries supply explicit raw or structured buffer views.
Group creation validates the native view range and layout's minimum size
before reserving any descriptors, including for dynamic storage bindings.

Fixed, fully bound WGSL compute buffer arrays can use ordinary layouts for
constant or uniform descriptor selectors. A divergent selector requires
nonuniform layout metadata. Admission uses the same Naga uniformity facts
as the HLSL writer.

WGSL pipeline compilation checks each active resource against the selected
stage's bind-group layout visibility. It checks scalar and array bindings
before native PSO creation or cache lookup.

Native compute Int64 atomics use Shader Model 6.6 or newer and require
Int64 shader operations. Raw and structured UAV buffers use the descriptor
heap atomic capability from OPTIONS11. Workgroup atomics require the
separate group-shared capability from OPTIONS9. Direct HLSL and single
compute DXIL can declare the corresponding operation and address space.
Callers provide correctly aligned offsets and record synchronization.

Indirect compute dispatch reuses one immutable command signature per
device. A WGSL pipeline that reads `num_workgroups` reuses its own
root-specific signature. This path also copies the 12-byte argument
record twice into a 24-byte temporary and restores the caller's indirect
argument state before execution. Ordinary indirect dispatch uses the
caller's argument buffer directly.

Native compute can bind a single-mip, non-array R64Uint texture for
atomic add and unsigned min/max. DX12 uses a `R32G32_UINT` resource and
UAV with a shader-declared `RWTexture2D<uint64_t>`. Shader Model 6.6, Int64
instructions, OPTIONS9 typed atomics, and typed format support qualify the
route. The pipeline compares the artifact's per-register atomic use in
PSV0 with explicit R64Uint atomic bindings. Optional copies preserve
eight-byte scalar texels. The texture has no sampled or attachment view.

Native compute R32Uint and R32Sint add, min/max, bitwise operations,
exchange, and compare-exchange use the corresponding format capability
bits. The caller declares the native shader's
operation requirements and the logical format of each UAV. Pipeline
creation verifies that each required format has a matching active typed 2D
UAV register span. PSV0 identifies the register and view kind. The shader
author supplies the operation and signedness contract for 32-bit atomics.

WGSL vertex and instance indices include the draw's starting offsets.
Shader Model 6.8 vertex programs add `SV_StartVertexLocation` and
`SV_StartInstanceLocation` to the corresponding native IDs. They use
the ordinary root signature and indirect draw path. Earlier profiles
apply the existing root constants to the same entry arguments, including
members of an input structure passed to a helper. The indirect-count route
query reports which pipeline needs argument adjustment.

WGSL uniform layout minimum sizes cover the declared type's byte span.
With the qualified `shader-f16` feature, uniforms containing halves lower
to ordinary CBV bindings with 16-byte rows. Loads use the WGSL member,
matrix, and array offsets and extract the addressed half from its DWORD.
Native UNIFORM allocations, including imported buffers, cover the
256-byte-aligned CBV range while callers bind the logical WGSL range.
Fixed uniform binding arrays retain their compute and descriptor-indexing
requirements.

WGSL shaders declaring `immediate_address_space` can use `var<immediate>`
with host-shareable 32-bit scalars, F16, vectors, matrices, and array-free
structures within 64 bytes. DX12 lowers accessible DWORDs to explicit
HLSL packing offsets and reconstructs values using WGSL member offsets and
matrix column strides. The selected stage's layout ranges must cover each
accessible DWORD. Callers write these ranges through the existing explicit
constants commands. With the separately qualified `shader-f16` feature,
each half is extracted at its declared byte offset. Halves in the same
DWORD are supplied together by the caller. Compatibility
`var<push_constant>` uses its own ABI.

Direct HLSL uses resource reflection from the selected DXC entry when the
layout has stage-restricted bindings. The check covers intersecting register
ranges, including arrays, before PSO creation. Explicit layouts still
provide view-type information that native reflection cannot infer.

For single-program DXIL with stage-restricted bindings, DX12 checks the
program kind and register ranges from the container's `PSV0` data. Compute,
raster, and mesh admission can perform these checks without a DXC runtime
or optional reflection data. Ray pipelines look up the selected DXIL library
export in RDAT and validate its name and shader kind before state-object
creation. DXR validates the native library's resource bindings.

An explicit native register map assigns every logical binding to its exact
register. The bind-group index supplies the register space and the binding
type supplies the register class. Different classes can share register
numbers. Same-class ranges in a space must be disjoint and must avoid the
root constant register when that register is in use. Descriptor table order
and dynamic offsets still follow logical bindings. The serialized root
signature carries the assignment into the PSO cache key.
Single-program HLSL and DXIL check that explicit maps cover their native
register spans at the selected stage. An unbounded range uses a layout
runtime capacity at the same class, space, and base register.

## Recording, submission, and completion

The normal run path is:

```text
create resources and pipelines
        -> begin encoder
        -> record ordered Core commands into a D3D12 list
        -> finish into retained recorded work
        -> preflight plan, execute lists, signal queue fence
        -> receive checkpoint/completion point
        -> poll or await its native fence
        -> retire retained owners after that fence completes
```

Recording applies explicit Core transitions, clears, copies, render passes,
bind groups, draw/dispatch operations, queries, indirect commands, and
secondary bundles to the command list. Secondary work is a D3D12 bundle with
its own allocator; the direct encoder retains it after `ExecuteBundle` so the
allocator, list, pipelines, descriptors, and resources survive execution.
Fixed-count indirect draws in a bundle reuse the single-operation `DRAW` or
`DRAW_INDEXED` signature from primary recording. The argument buffer also
retains its device's signature cache. The calling plan transitions that buffer
to `INDIRECT_READ` before executing the bundle.
Bundle draws inherit the primary blend constant and stencil references until
the secondary work sets them. After the bundle, the primary encoder restores
its saved values, including defaults. Independent front and back references
use the same OPTIONS14 and command-list interface checks as primary recording.

Submission preflight validates the packet before native handoff. It separates
direct and compute segments, issues caller-specified native queue waits, and
signals the appropriate queue fence. A checkpoint covers only the ordered
prefix on that lane. A direct-queue fence does not prove compute completion
unless the submitted plan explicitly joined the queues.

Recorded work retains command allocators, upload resources, descriptor owners,
pipeline and resource references, query bookkeeping, and other native
dependencies in an in-flight list keyed by its fence value. `completion_status`
reads `ID3D12Fence::GetCompletedValue`; only a completed native fence releases
those owners. CPU waiting uses an event registered with that fence instead of
busy waiting. Bounded waits use the same Windows thread-pool registration
with a relative timeout. When the executor observes expiration it checks the
fence again, giving an observed completion or device loss precedence. The
event remains owned until the native fence can no longer signal it. A value
of `u64::MAX` represents device removal or indeterminate accepted work, so
it is quarantined rather than treated as a canceled packet.
If a later native handoff fails, work covered by a signal already issued on
its lane remains eligible for fence retirement. Work without a covering
signal stays in quarantine. The packet still reports an indeterminate
handoff when the failed operation leaves its exact accepted prefix unknown.

Before issuing a step, DX12 checks that its present lease is still current
and that each newly active lane can receive a fence value. A clean failure
returns the untouched suffix. When earlier steps issued, DX12 signals their
dirty lanes and returns a covering checkpoint for every accepted lane.
Ordinary queue resume verifies those fences and executes only the suffix.
Multi-queue packets return one-shot waits for untouched steps in their
diagnostic tail. Native handoffs with uncertain effects retain their owners
through the existing quarantine path.

An indeterminate submit produces a DX12 failure report with the failing
operation, lane, and step when applicable. It reports already-issued step
payloads and the exact checkpoints of successfully signaled lane prefixes.
These checkpoints can be observed independently of the failed operation.

New primary and secondary encoders, idle waits, and packet admission check
the native device removal status before allocating command objects or
handing work to a queue. A removal observed after admission still follows
the native handoff and fence rules above.

DXR in-place refits use the same structure address for source and destination.
Submission checks the opted-in build signature against earlier native
handoffs and earlier compatible commands in the current packet. The update
preserves geometry layout or instance count and retains both prior and new
TLAS dependencies through the enclosing fence. Renderer-supplied barriers
and queue waits remain required. Indexed BLAS refits are deferred pending a
contract that preserves index contents.

DXR compaction uses a private size output and readback. The ticket observes
the fence for its submitted lane and carries the identity of the specific
source build or refit. Submission rechecks that identity before issuing the
compact copy. A compacted TLAS carries its source's BLAS owners and refit
signature. Callers still specify barriers and queue ordering for traversal.

## Surface lifecycle

`Dx12::surface_from_raw_window_handle` accepts only a Win32 handle and carries
the safety requirement that the host keeps it alive while the surface and all
image leases exist. Configuration creates or replaces a DXGI flip-model
swapchain and increments the surface generation. The surface permits one
exclusive acquired image at a time; an unpresented lease clears acquisition on
drop. Resize or reconfiguration makes old-generation images stale, and a
handoff validates generation before calling `Present`.
Recorded attachment and resolve commands retain their backbuffer and RTV
heap until abandonment or native fence retirement. Reconfiguration retires
completed lists and checks only the current swapchain generation's owners.

The surface supports its declared formats, FIFO presentation, and immediate
presentation only where DXGI reports tearing support. It owns swapchain state,
not an application frame loop, fullscreen policy, or host event processing.
The configured format describes the render-target view. An sRGB RGBA8 or
BGRA8 configuration uses UNORM flip-model storage and an sRGB render-target
view. The raster pipeline matches the leased view format. Render-pass resolve
still requires compatibility with the underlying typed swapchain resource.

HDR configuration admits scRGB with `Rgba16Float` and HDR10 with
`Rgb10a2Unorm` on a monitor with active HDR. The final swapchain checks
native presentation support before setting its color space and publishing
the new generation. The caller can query the current monitor and reconfigure
after movement or a change to the system HDR setting. Output snapshots report
current DXGI color and luminance facts separately from the selected configuration.

## Capability boundaries

Timestamp queries can be recorded on the graphics or compute queue, between
encoder commands or at a chosen position in a raster or compute pass. A
pass-local timestamp measures the position after preceding commands,
including any already recorded pass setup, and uses the same slot tracking
and resolve rules as encoder-level timestamps.
Resolved values are raw ticks. Core can convert two samples from the
same lane to nanoseconds using the native timestamp scale and counter
width, including a single wrap shorter than the clock's full period.

An occlusion query begins and ends in the same DX12 raster pass. If the
pass declares a query set, its occlusion slots come from that set.
An indeterminate handoff quarantines the affected query slots and
preserves the ordering facts for other slots in the same heap.

WGSL layout inference reflects active globals for selected vertex, fragment
or compute entries. Core combines compatible declarations before applying the
ordinary layout admission checks. Runtime arrays require a supplied capacity.
Inferred arrays allow divergent indexing where supported, and float textures
sampled with samplers use a filterable default. Callers can specify those
binding policies directly with explicit layouts.

Direct HLSL uses reflection of the selected entry compiled with its pipeline
constants. DXIL inference takes a single declared entry and checks the native
shader stage. Both preserve the binding register and space. Native inference
covers constant buffers, raw and structured storage buffers, samplers,
integer sampled textures and acceleration structures. A reflected storage
buffer with `min_size = 0` specifies no inferred minimum range. Bind-group
creation still requires a nonempty raw or structured view with a valid
native range. Samplers default to filtering. Float and depth sampled textures,
typed UAVs, counter buffers, and native constants that share the immediate
register require explicit layouts. Native register overlaps that cannot fit
the Core binding namespace require distinct shader registers.

Raster fragment shaders can require the rasterizer-assigned primitive index.
WGSL lowers the declared builtin to `SV_PrimitiveID`; direct HLSL and DXIL
declare the same requirement. DX12 mesh fragments that declare this portable
requirement are rejected because their primitive ID would be a mesh shader
output chosen by the shader author.

For forced early depth and stencil tests, WGSL fragment entries lower
`@early_depth_test(force)` to HLSL `[earlydepthstencil]`. HLSL and DXIL
callers declare the requirement for their native attribute or shader flag.
WGSL conservative depth modes require distinct depth-output semantics and
are currently rejected at shader creation.

Native fragment barycentrics require shader model 6.1 and a successful
`D3D12_OPTIONS3.BarycentricsSupported` query. The declared requirement is
admitted for HLSL profiles at least 6.1 and for direct DXIL. The WGSL route
has no barycentric builtin.

Vertex and fragment WGSL shaders can declare `view-index` on a device
with view instancing and Shader Model 6.1. The HLSL interface takes an
unsigned `SV_ViewID` and converts it to the signed WGSL builtin at entry.
The same requirement qualifies native HLSL and DXIL declarations. Native
view IDs follow the pipeline's view ordinals.
The pass mask selects a subset of those ordinals. All enabled ordinals
must fit the pipeline and the attachment array views. Primary and
secondary draws use the same admission rule, and all views use the
pass's shared viewport and scissor.

Mesh multiview requires both native mesh shaders and view instancing.
DXIL mesh pipelines use the same view ordinal mapping and mask admission
as vertex raster pipelines. Each view location uses its ordinal as the array
index within the attachment view and selects viewport zero. A meshlet
instance selects scene geometry independently of this mapping.
Verify a sparse view mask by reading back
each enabled layer and a clear sentinel in the skipped layer.
The mesh pipeline cache includes the view count and any per-face stencil
masks. The native runtime checks shader output and shared-memory requirements
for the selected view count.

WGSL fragment sample index and sample interpolation lower to
`SV_SampleIndex` and the HLSL `sample` modifier. They select native
sample-frequency execution. Pipeline and attachment sample counts still
follow the ordinary multisample format checks.

Per-face dynamic stencil references require the DX12 OPTIONS14 feature
and the graphics command list 8 entry point. Raster state restoration
replays the front and back references after internal draws or bundles.
Independent static masks use the same OPTIONS14 feature. Vertex raster
and mesh pipelines place the two mask pairs in the native depth-stencil 2
stream subobject. The pairs are part of the pipeline cache key and stencil
write admission considers each face's operations and write mask.

The `subgroups` shader requirement uses the device's Shader Model 6 wave
support and its reported minimum and maximum wave widths. WGSL admits
subgroup lane index, width, vote, ballot, arithmetic, shuffle, and quad
operations in compute and fragment entries. WGSL subgroup ID, subgroup
count, and subgroup barriers have no exact lowering on this route.
Native HLSL and DXIL follow the operation and stage rules of their profiles.
The explicit fixed-size compute path additionally requires Shader Model 6.6.

WGSL shaders declaring `push-constants` may use a `var<push_constant>`
block for vertex, fragment, or compute. A flat structure of 32-bit
integer or float scalars and vectors lowers to the existing root constant
register when no member crosses a 16-byte HLSL constant-buffer row. The
layout ranges for each stage using the block must cover its full byte span.
Arrays, matrices, nested structures, and narrower member types remain
outside this layout path.

WGSL composite pointer access uses Naga's standard member and index
lowering for pointers accepted by its frontend. The WGSL shader route
can explicitly qualify the `pointer_composite_access` requirement.
The vendored frontend rejects `unrestricted_pointer_parameters`.

For WGSL packed 4×8 integer dot products, DX12 selects the native signed
or unsigned HLSL intrinsic when the device supports Shader Model 6.4.
The Shader Model 6.0 route keeps the equivalent integer lowering. Naga and
DXC use the same selected shader model.

Native F16 declarations require Shader Model 6.2 and
`D3D12_OPTIONS4.Native16BitShaderOpsSupported`. DXC receives
`-enable-16bit-types` for declared generated and direct HLSL. DXIL
callers declare their bytecode's use of native F16. WGSL can use F16
arithmetic, workgroup data, raw storage, shader interfaces and finite
F16 overrides. WGSL uniform and push constant layouts containing F16
are rejected because the current HLSL constant layout cannot express
all two-byte member offsets.

Native shader F64 requires Shader Model 6 and
`D3D12_OPTIONS.DoublePrecisionFloatShaderOps`. HLSL and DXIL declarations
enable the double-precision shader route. The fact covers shader operations
and native buffer access subject to the shader profile and exact resource
description. Texture pixel formats and Input Assembler formats have their
own capabilities.

Native HLSL and DXIL signed and unsigned 64-bit computation require
Shader Model 6 and `D3D12_OPTIONS1.Int64ShaderOps`. Atomic operations and
64-bit resource formats keep separate admission.

Native HLSL and DXIL can declare exact 16-bit integer types with the
Shader Model 6.2 and OPTIONS4 native 16-bit gate. HLSL requests the
native DXC type option for `int16_t` and `uint16_t`. WGSL currently
cannot declare either integer type.

The implemented public route includes graphics and compute, multiple render
targets, indirect drawing and dispatch, queries, secondary recording,
standalone clears and resolves, placed memory, scoped native/external interop,
and pipeline caches. Raster supports up to eight color attachments,
per-target blend and write masks, depth/stencil and read-only depth/stencil
views, attachment clears, and resolves. Formats, sample counts, shader model,
resource descriptors, and device tiers remain part of admission.

Descriptor arrays and indexing are constrained by device capabilities,
descriptor-table layout, and shader route. The verified WGSL non-uniform
uniform-buffer and storage-buffer routes are fixed, fully bound, compute-only
shapes. They do not promise runtime or partially bound arrays, dynamic offsets,
other stages, or mismatched shader/layout metadata.

BC1–BC7 compressed textures use DXGI mappings with checked creation,
block-aligned upload/copy, and sampling. `texture_support` remains the source
for a particular format, dimension, and usage. ETC2/EAC and ASTC have no DXGI
mapping in this backend.

Ray tracing, ray query, mesh shaders, sparse resources, dynamic raster state,
texture atomics, GPU-counted draws, multiview, and source wave-size selection
are individually gated. They are available only when the selected adapter and
the exact descriptor, pipeline, and shader path qualify.

## Non-goals

This crate does not implement RenderGraph scheduling, global resource-state
tracking, CPU command replay, generic device-loss notification policy,
arbitrary shader format conversion, or a DXR state-object cache. Open
cross-backend work and its reopening conditions are recorded in
[`TODO.md`](../../TODO.md).
