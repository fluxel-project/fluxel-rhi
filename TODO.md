# Open Design TODOs

This is the authoritative list of unfinished RHI work selected for follow-up.
An item stays open until its stated evidence and acceptance conditions are met.
Unselected capabilities below are not an automatic implementation backlog.

## Active items

### DX12 device limit verification

**Status:** Raster and mesh pipelines and raster pass recording use the
published color attachment slot limit. Empty slots consume indices.
The DX12 backend uses the D3D12 render target and view instance ceilings
at native recording and creation boundaries.

**Acceptance:** Check an exactly full sparse color target list and an
over-limit list in raster and mesh creation and pass recording. For
multiview, check masks that reach the highest supported view and masks
with a bit above it. Include the prepared raster path used by queries.

**Compute dispatch:** Core checks each direct dispatch count against the
published axis limit before recording. Verify exact-limit and one-past-limit
counts and confirm that a valid dispatch can follow the rejected call.
GPU-produced indirect arguments must obey the same native bounds.

**Push constant range:** Match layout support and creation at the published
byte ceiling and one DWORD beyond it. Include a short range starting near
the ceiling and a legal byte span that exceeds the combined DX12 root
signature budget once the descriptor tables and root CBVs are counted.

**Vertex input:** Check slot and input-element limits independently of the
compiled vertex signature. Microsoft Learn's input-element reference lists
slots 0 through 15 while the SDK defines 32 IA vertex input slots. Verify
slot 15, slot 16, slot 31 and the first rejected slot on a real DX12 device.
Use an explicit Core offset equal to `D3D12_APPEND_ALIGNED_ELEMENT` as a
negative case. It must be rejected before PSO creation.
Verify duplicate shader input locations are rejected across two buffer slots
and within one slot. Check native signature linkage for the selected artifact
and cover zero-stride constant input independently of ordinary vertex stride.

**Storage buffer views:** The exact-range support query, initial binding
and dynamic rebinding share the DX12 descriptor admission path. Verify
raw offsets at 4 and 16 bytes, a four-byte view length, and structured
view offsets and lengths at stride boundaries. Cover excessive element
counts, writable upload views, foreign buffers, and dynamic ranges that
overflow or extend past the backing buffer.

**Query resolve support:** Match the DX12 support query with recording for
timestamps on direct and compute lanes and for occlusion and statistics on
the direct lane. Check a foreign lane, slot bounds, offset alignment, wrong
usage and the last byte of the destination. Producer ordering and active
query state are verified during command execution.

**Texture mip support:** Match the DX12 reported mip ceiling to creation
for single-sample and multisampled textures of the same extent. Check a
non-power-of-two extent and verify multisampling reports and admits one
mip while rejecting a second mip.
Check the native `MIP` flag against the actual allocation format, including
typeless allocations used for reinterpreted color and sampled depth. When
the device reports no mip support, the one-level descriptor must report a
ceiling of one and the multi-level request must fail before allocation.
For multisampled color attachments, compare texture and view support with
creation for the base and reinterpreted formats. Check native multisampled
render target support and the requested sample count for the view format.
For sampled texture layouts, reject multisampled 1D, 3D, cube and cube-array
bindings in both support and creation. Keep 2D and 2D-array multisampled
layouts and single-sample cube layouts covered by positive cases.
For cube views, compare query and creation for the typed SRV format's native
cube support. Read back samples from selected depth cube faces and from a
cube-array view with a nonzero first face. Cover the zero-based single cube
and a depth/stencil view that requests both aspects as a sampled cube.

**Pipeline attachment support:** Compare the DX12 query with raster and mesh
pipeline creation for exact multisample counts across all occupied color slots
and the depth/stencil target. Include a format that rejects the requested count,
unsupported blending, an oversized sample mask, and alpha-to-coverage with one
sample. Use compatible shaders so the comparison isolates attachment admission.

**Raster pipeline support:** Compare descriptor support with ordinary raster
pipeline creation for same-device inputs, invalid strip state, multiview
limits, vertex input formats and offsets, and missing or wrong-stage shader
entries. Verify native register overlap and DXIL specialization reject in the
shared descriptor path. Confirm signature linkage and specialized shader
validation still run during creation.

**Mesh pipeline support:** Compare the ordinary descriptor query with mesh
pipeline creation for same-device inputs, supported line and triangle output,
invalid strip state, attachment admission, and declared mesh and task sizes.
Cover wrong-stage entries, non-DXIL stages, native register overlap, and a
fragment that requests the rasterizer primitive index. Native creation still
validates the linked DXIL artifacts.

**Compute pipeline support:** Compare the descriptor query and creation for a
known compute entry, missing or non-compute entries, foreign shader, layout or
cache objects, invalid declared workgroup sizes, and required subgroup sizes.
Exercise native register overlap and DXIL specialization rejection. Verify that
the query uses existing shader metadata and leaves specialization, compilation,
and cache or PSO creation to the full pipeline path.

**Index buffer views:** Verify that DX12 support, direct recording and
bundles agree on INDEX usage, element alignment, byte range, and the native
32-bit view size. Include an aligned Uint16 range whose start is rejected
for Uint32 and a valid range ending at the buffer boundary.

**Ray dispatch support:** Match the DX12 support query with trace recording
for valid shader tables and for missing SHADER_TABLE usage, misaligned
sections, excessive stride, out-of-bounds ranges, and excessive dimensions.
Verify raygen requires one record and a failed optional section leaves the
encoder free of new buffer owners. Record and read back a supported dispatch.
DX12 allows a zero dispatch axis and launches no threads. Verify the table
and bound-pipeline rules still apply and that the output sentinel survives.

**Ray pipeline support:** Compare the descriptor query with DXR state-object
creation for declared exports and general, triangle, and procedural hit groups.
Cover foreign inputs, an acceleration-only device, wrong-stage group imports,
non-DXIL stages, local constants above the root budget, unsupported cache,
recursion and attribute limits, and native register overlap. Creation still
verifies DXIL export and root signature linkage.

**Pipeline admission after device removal:** On an isolated removed DX12
device, check that valid raster, mesh, compute and ray descriptors return
`DeviceLost` both through support and creation. Also cover attachment and
vertex format support. The active-device check precedes feature probes or
artifact creation and runs once per pipeline admission.

**Descriptor admission after device removal:** On an isolated removed DX12
device, verify texture view and sampler support and creation return
`DeviceLost`, including explicit usage, sampled mapping and a 3D slice view.
Verify bind-group creation rejects before reserving descriptor slots. Use
previously created resources and populated descriptor pages to exercise the
reuse path that does not need to allocate a new native heap.
Check that ordinary buffer and texture creation distinguish device removal
from unsupported descriptors before entering native allocation. Verify
bind-group and pipeline layout support and creation on the same removed
device, including an empty layout that otherwise requires no format probes.

**Counted indirect draw support:** Compare the query with DX12 recording
for indexed and non-indexed argument strides and checked offsets, including
shared argument/count storage and a zero maximum. Verify the reported direct,
copy and pack routes on their corresponding pipelines and batch sizes.

### DX12 explicit memory admission

**Acceptance:** Compare explicit heap support with creation for valid heap
classes, unknown classes, zero sizes, native alignment overflow, and mixed
heaps on Resource Heap Tier 1 and Tier 2 devices. Include a non-aligned heap
size and verify placement stays within its requested byte extent. Match the
texture allocation requirements and placement paths for supported shapes,
transient attachments, and removed devices.

Verify that reserved 3D textures require Tiled Resources Tier 3 and reject
oversized extents on each axis before allocation. Cover valid 2D arrays and 3D
textures, the packed array tail restriction below Tier 4, and creation and
allocation on a removed device.

Exercise an entirely packed sparse texture on hardware that reports a zero
standard tile shape. Check that the reported grid is empty, mip zero begins
the packed tail, and a full-tail bind permits a write and readback. Verify that
unmapping the tail and mapping a Tier 4 array slice affect only the requested
range. Below Tier 4, verify that an array with a fully packed single-slice
probe receives a structured unsupported error.

**Rejected for DX12 reserved textures:** `Depth24PlusStencil8` and
`Depth32FloatStencil8`. The [D3D12 DXGI format tables](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/hardware-support-for-direct3d-12-1-formats)
list no tiled resource support for the R24G8 and R32G8X24 depth/stencil
families. Texture creation rejects these sparse descriptors before native
format probes. Revisit if a native capability explicitly supports those
families and specifies how to address their depth and stencil tile mappings.
In Stage 8, check the rejection with attachment and sampled usages.

### DX12 explicit heap residency

**Acceptance gate.** In Stage 8, write a pattern to a resource placed in an
explicit default heap, wait for its last uses on all lanes, evict the heap,
make it resident again, and read the same bytes. Check the initial reference,
balanced reference operations, wrong-device and removed-device errors, and the
support result for CPU-visible heaps.
For sparse allocation residency, include both heap classes, verify the support
gate on devices without tiled resources, and evict and restore a heap while
its completed tile mappings remain installed. Verify data through a mapped
tile after restoration and repeat with a mapping on another queue lane.
For asynchronous residency, cover both budget modes, independent completion
for two enqueues, receipt and wait cancellation, registration failure after
native acceptance, and device loss. Verify that only native residency fence
completion admits subsequent host-gated work and that neither queue's
submission completion advances due to residency.
Cover explicit residency waits on direct and compute, including queue use
before a host wait, the same point on both lanes, and a direct-lane wait
followed by sparse mapping. Verify order on both sides of the encoder
boundary, wrong-device and unsupported devices, and fence retention across
partial acceptance and cancellation of unsubmitted work.

**Pending native paths.** CPU-visible heaps require an ordering contract for
active CPU mappings in addition to GPU completion. Keep their residency
operations gated until both are covered.
Resolve physical owner identity for committed resources before exposing their
residency operations. Placed resources use the existing heap operations.

### DX12 structured shader diagnostics

**Status:** WGSL parse and validation errors preserve compiler messages and
source byte ranges, line and column when available.
DXC failure reports preserve location, severity, and the raw report for
direct HLSL and generated HLSL. Successful raster and compute pipelines
retain DXC compilation diagnostics per stage and entry.

**Acceptance:** Exercise an invalid WGSL source with multiple diagnostic
labels and verify UTF-8 spans and the original rendered report.
Compile failing HLSL and verify location, severity, and fallback for
unlocated diagnostics. Verify generated-source diagnostics are identified
as HLSL positions rather than mapped to the input WGSL.
Create a valid raster or compute pipeline that emits a DXC warning and
verify the warning remains queryable, including when using a PSO cache.

### Core pipeline layout inspection

**Status:** Core-created raster, compute, mesh and ray pipelines retain the
portable pipeline layout. Layout metadata exposes the original bind groups,
entries and push-constant ranges.
DX12 can infer portable layouts from selected WGSL raster and compute
entries, including explicit capacities for runtime binding arrays. HLSL and
DXIL inference remain open.

**Acceptance:** Query cloned pipelines and reconstruct compatible bind groups
from the returned metadata. Check that native imports omit unavailable
metadata and absent indices return no bind-group layout.
Build and execute raster and compute pipelines with inferred layouts. Check
shared bindings across stages, omitted inactive resources, group gaps,
filterability defaults, array capacities, and conflicting declarations.

### DX12 runtime and partially bound TLAS arrays

**Status:** DXR devices with descriptor indexing support runtime and partially
bound TLAS descriptor tables. Holes use the native null acceleration view.
Resource admission rejects bottom-level structures before descriptor writes.

**Acceptance:** Read back distinguishable results from multiple populated TLAS
slots for fixed and runtime arrays, including nonuniform indexing and holes.
Verify capacity and shader layout rejection, and TLAS/BLAS retention through
submission completion. Qualify each supported shader route separately.

### DX12 secondary-source alpha blending

**Status:** Both secondary-source alpha factors lower to their native DX12
blend factors and share the single-target dual-source admission check.

**Acceptance:** Read back an RGB result that distinguishes secondary-source
alpha from secondary-source color. Cover both factors and invalid MRT use.

### DX12 whole-pass statistics semantics

DX12 exposes a native interval total for pipeline statistics on direct command
lists through `QueryKind::PipelineStatistics`. Core's `PassStatistics`
requires stage-boundary snapshots. Keep the DX12 boundary domain unavailable
and verify the direct interval query and compute-lane rejection separately.
Reopen boundary sampling when a native route can report each required sample.

### DX12 vertex input format admission

**Status:** Input layout creation and vertex format queries share the D3D12
input-assembler format check. Packed RGB10A2 UNORM and BGRA8 UNORM are mapped
when the device reports support for those vertex formats.

**Acceptance:** Read back normalized and swizzled vertex data from an actual
raster draw. Check query and pipeline agreement, including early rejection
for unsupported packed and 64-bit formats.

### DX12 mesh output topology

**Status:** Mesh pipelines and topology queries share admission for native
line-list and triangle-list output. Point and strip output are unavailable.
DX12 reports the mesh and task shader limits on mesh-capable devices.

**Acceptance:** Verify line and triangle output on a supporting GPU and
reject unsupported topologies and shader/PSO topology mismatches.
Check shader declaration admission against the reported invocation, output,
and combined task payload limits.
Check the optional Mesh and Task workgroup declarations at exactly 128
invocations and with zero, excessive or overflowing dimensions. DXIL without
a declaration still uses native PSO validation. Signature packing, dynamic
output counts and payload linkage require native artifact validation.

DX12 publishes the shader-written render target array index bound. Tier 1
mesh devices report eight choices unless OPTIONS9 proves the full 2048
layer range. A DXIL mesh artifact that can select index 8 or higher declares
`SHADER_FEATURE_DX12_MESH_FULL_ARRAY_INDEX`. Verify rejection on devices
without that fact and distinct writes to layers 7 and 8 where supported.
The caller bounds shader indices to the actual view. Cover multiview separately.

Verify the rasterizer-assigned primitive index in ordinary DX12 raster
pipelines with WGSL, HLSL and DXIL fragment stages. Mesh pipelines reject
that portable requirement. Reopen a portable mesh primitive ID only when a
consumer defines the producer and consumer contract for a shader-written ID.

### DX12 forced early depth and stencil

**Status:** DX12 supports the forced test mode for WGSL and declarations of
the corresponding native HLSL attribute and DXIL flag.

**Acceptance:** Verify that depth and stencil rejection suppresses fragment
storage writes. Cover depth and stencil updates when the fragment later
discards, and reject nonfragment requirement declarations. WGSL conservative
depth remains pending until its depth-output and interpolation rules are
mapped per mode and verified with depth and storage readback.

### DX12 WGSL inline constants

**Status:** The compatibility `var<push_constant>` WGSL route in compute and
raster uses the existing root-constant ABI for flat 32-bit scalar and vector
layouts. Pipeline admission checks the full block span against that stage's
ranges. Standard `var<immediate>` is pending frontend and layout support.
Its store type must be host-shareable and constructible. At shader creation,
reject direct arrays and structures containing an array at any nesting depth.

**Acceptance:** Verify byte-exact readback for adjacent and padded members,
partial writes, stages with different ranges, and rejection of missing
coverage and reserved register overlaps. For the standard route, cover legal
matrices and array-free nested structures, plus negative cases for a direct
fixed-size array, an array member, and an array in a nested member. Evaluate
array lowering separately for native HLSL/DXIL or compatibility push constants
only where the root-constant ABI preserves offsets and indexing.

### DX12 WGSL composite pointer access

**Status:** The WGSL shader route qualifies composite pointer access and
the existing HLSL lowering consumes the resulting member or index access.
The vendored frontend does not implement unrestricted pointer parameters.

**Acceptance:** Verify implicit member and array dereference with a GPU
readback, including storage access through a pointer expression. Reopen
unrestricted arguments when frontend validation and HLSL lowering can
preserve resource identity, offsets and aliasing semantics across calls.

### DX12 native 16-bit integer shaders

**Status:** HLSL and DXIL use the Core ShaderI16 device requirement and
the DX12 native 16-bit probe. WGSL has no matching integer types.

**Acceptance:** Verify signed and unsigned 16-bit arithmetic and packed
buffer reads and writes, including HLSL reflection with the native type
option and rejection below Shader Model 6.2. Reopen WGSL only when a
standardized WGSL extension defines the exact integer types, their memory
layout, validation, and target availability.

### DX12 native 64-bit integer shaders

**Status:** Core device requests and DX12 HLSL and DXIL declarations
qualify signed and unsigned 64-bit computation with native Int64ShaderOps.
Int64 atomics and shader interfaces have separate contracts. Standard WGSL
currently has no 64-bit integer scalar route, so WGSL lowering is unavailable.

**Acceptance:** Verify wide signed and unsigned arithmetic and buffer
readback with values exceeding 32 bits, including device and profile rejection.
Do not expose these native operations through WGSL unless a standardized
64-bit scalar route, frontend support, exact lowering, and boundary-value
evidence become available.

### DX12 native double-precision shaders

**Status:** HLSL and DXIL shader declarations and Core device requests use
native DoublePrecisionFloatShaderOps. Standard WGSL has no 64-bit scalar
route. Raster interface forms have separate lowering and signature constraints.

**Acceptance:** Verify feature and shader admission plus double arithmetic
and raw buffer readback with values that distinguish F64 from F32. Keep
native vertex formats and texture support independently qualified.
Do not admit WGSL 64-bit bitcasts, buffer layouts, or stage signatures unless
a standardized scalar route and exact lowering exist.

### DX12 native half-precision shader values

**Status:** DX12 qualifies native F16 with Shader Model 6.2 and OPTIONS4.
WGSL and HLSL use the native DXC type option and direct DXIL can declare
the device requirement. WGSL F16 uniform and immediate layouts are a planned
D4 ABI task. Each shape remains unavailable until standardized source syntax,
byte layout, and DX12 lowering are verified.

**Acceptance:** Verify WGSL arithmetic, storage, workgroup data and raster
interfaces with exact half values and readback. Cover finite F16 overrides,
invalid scalar widths, unsupported devices and insufficient HLSL profiles.
Admit uniform and immediate layouts only when lowering preserves two-byte
offsets and the declared aggregate layout.

### DX12 packed 4×8 integer dot products

**Status:** WGSL raster and compute pipelines select the native intrinsic
on Shader Model 6.4 devices and preserve the software lowering on older
Shader Model 6 devices.

**Acceptance:** Compare both compiler routes and GPU results for signed
and unsigned packed operands, including negative bytes and maximum
unsigned values.

### DX12 subgroup shader admission

**Status:** DX12 admits the declared wave route from the device's Shader
Model 6 and OPTIONS1 wave support. WGSL compute and fragment entries can
use width, lane index, vote, ballot, arithmetic, shuffle and quad operations.
WGSL subgroup ID, subgroup count and subgroup barrier require verification of
an exact standardized-language and native mapping before they can be admitted.

**Acceptance:** Verify supported WGSL operations with compute and fragment
readback, including partial waves and fixed-width compute variants where
supported. Verify WGSL stage and unsupported builtin rejection. Reopen
subgroup ID, count or barrier after an exact native or lowered mapping can
be verified across different wave configurations.

### DX12 pass-local timestamps

**Status:** Graphics and compute passes can record timestamps at an
explicit command position. Encoder timestamps, pass-local timestamps and
resolves share query slot and native lane ordering rules.

**Acceptance:** Write before, between and after pass commands, resolve the
slots, and confirm their ticks on the producer lane. Cover graphics and
compute queues, invalid slots and reset and reuse.
Verify normalized durations use only the valid counter bits, handle one
wrap, and reject the reserved invalid value before counter arithmetic.

### DX12 WGSL sample-frequency shading

**Status:** WGSL validation admits fragment sample index and sample
interpolation. HLSL uses native sample-frequency semantics.

**Acceptance:** Read distinct per-sample fragment results from a
multisampled target and verify interpolation at sample positions.
Cover the sample index and interpolation triggers separately.

### DX12 per-face stencil references

**Status:** Core raster passes can set independent dynamic front and
back stencil references on devices that report the DX12 OPTIONS14
feature. Graphics state restoration retains both values. Raster and mesh
pipelines also admit independent static read and write masks.

**Acceptance:** Render opposite-facing primitives against distinct
references. Verify state restoration after internal draws and bundles,
setting a shared value after a split value, and unsupported-device
rejection. Verify independent static masks, read-only attachment admission,
and cache separation for distinct masks. Include mesh and multiview on
devices that support the respective combinations.

### DX12 shader view index

**Status:** WGSL vertex and fragment entries can read the signed view
index through the native Shader Model 6.1 unsigned view ID. Native
HLSL and DXIL declarations share device admission.

**Acceptance:** Read view-dependent output from nonzero attachment
layers with both vertex and fragment view-index inputs. Cover scalar
and struct input declarations, sparse masks, and incapable devices.
Reuse a four-view pipeline with masks selecting only the low ordinals
and nonadjacent views. Cover bundles and shared viewport and scissor
restoration.

### DX12 mesh multiview

**Status:** Mesh pipelines using DXIL admit native view instancing when
both device capabilities are present. Mesh shader view IDs use the native
artifact declaration. Raster masks and attachment layers follow the existing
view ordinal contract.

**Acceptance:** Verify distinct mesh and fragment results in enabled array
layers, preserving masked layers. Cover task amplification, direct and
indirect dispatch, depth attachments, and per-face stencil combinations
where supported. Check PSO rejection for excess view-dependent output.

### DX12 native fragment barycentrics

**Status:** The device capability and HLSL and DXIL shader routes admit
native barycentrics. The device must expose shader model 6.1 and report
native barycentric support. WGSL remains unavailable until the frontend and
lowering support a specified fragment barycentric builtin.

**Acceptance:** Compare perspective and linear barycentric weights from
ordinary and mesh triangles with readback. Verify HLSL profile, fragment
stage, source dialect and device capability rejection.

### DX12 acceleration structure refit

**Status:** TLAS and non-indexed BLAS refit use DXR in-place updates. Ordered
submission admission checks the prior build signature and the original
update opt-in. Indexed BLAS refit is a D6 design task. It requires explicit
immutable input evidence or a backend-owned snapshot retained from the first
build through every update.

**Acceptance:** Verify changed TLAS transforms and BLAS bounds by traversal
readback. Cover build and update in one packet, repeated and separately
submitted updates, a rejected update after an incompatible rebuild, and
retention of old and replacement TLAS dependencies. Check that indexed refit
is rejected before native recording while indexed builds remain available until
its design task is complete. Any snapshot path accounts for its copy, barriers,
memory overhead, and retention without implicit graph scheduling or
synchronization.
Verify triangle vertex layout queries and traversal for the added DXR SNORM,
UNORM and tightly packed three-component formats. Verify Tier 1.0 rejects the
formats that require Tier 1.1.
Verify packed RGB10A2 UNORM positions use their required four-byte stride
under DXR Tier 1.1.
Verify geometry-to-BLAS transforms by traversal readback, including a refit
with replacement transform data. Reject misaligned, undersized and foreign
transform buffers, and refits that change transform presence.
Verify the native build-size output for BLAS and TLAS after a covering
completion point. Compare it with the prebuild allocation bound and reject
invalid offsets, capacity, or conflicting buffer roles.
Verify compaction-enabled prebuild bounds and read back the compacted byte
size for BLAS and TLAS. Compact into an exact-sized destination and verify
traversal after dropping the source, including TLAS dependency retention.
Reject unsubmitted or pending tickets and reject tickets invalidated by a
rebuild between compact recording and submission.
Repeat the size query while refitting a compaction-enabled build and verify
that submission rejects a refit with mismatched original build flags.

**Planned GPU-generated TLAS instances:** The native DXR instance record
embeds a BLAS address. Keep native addresses in the backend and retain typed,
same-device BLAS dependencies through submission and TLAS use. Define explicit
logical GPU records indexed into a typed BLAS table, invalid-index handling,
and reporting. The caller or Renderer supplies producer-to-build ordering and
completion coverage. DX12 packs and validates native records without implicit
synchronization or lifetime policy. Acceptance proves table lifetime, ordering,
and update/refit semantics.

### DX12 acceleration structure clone

**Status:** The DX12 compaction path already copies a built acceleration
structure using a size ticket. Direct duplication into independent storage
has no corresponding Core/SPI command. The pinned wgpu DX12 backend uses
`CopyRaytracingAccelerationStructure` for both clone and compact.

**Acceptance:** Define an explicit same-device source and destination,
destination capacity, supported BLAS/TLAS shapes, and required barriers.
Keep referenced BLAS owners alive through TLAS use. Reject invalid aliasing
and undersized storage before native recording. Trace the destination after
releasing the source handle and compare hit/miss results with the original.

### DX12 conservative rasterization

**Status:** Device capability reporting and graphics/mesh pipeline admission
use the D3D12 conservative rasterization feature and filled triangle rules.

**Acceptance:** Verify coverage on a supporting GPU. Check early rejection
for point/line topology, wireframe, and a device without support.

### DX12 surface attachment formats and HDR

**Status:** Configuration and acquisition retain the selected attachment view
format, including sRGB views of supported UNORM flip buffers.
DX12 also exposes current monitor color and luminance facts.
The HDR routes qualify scRGB and HDR10 against the current output and native
swapchain. Their real display acceptance is pending.
The output snapshot includes the system SDR white level when the active
display paths identify an unambiguous target value.
HDR10 metadata write and clear are a planned opt-in narrow DXGI path. A
successful call means DXGI accepted the metadata. It does not establish that
the display received it.

**Acceptance:** Reconfigure a live HWND between linear and sRGB views and
observe the format and generation on the acquired images. Exercise matching
raster pipelines and verify sRGB encoding on a real presentation path.
Check SDR configuration suggestions against the exact configuration query,
including immediate mode with tearing and FIFO fallback without tearing.
Compare output snapshots across a window move between distinct displays.
Verify HDR10 and scRGB output on an HDR-active monitor, early rejection of
incorrect format/color-space pairs and HDR on SDR, and reconfiguration after
an output move or switch to SDR.
Verify HDR10 metadata write and clear across output changes and resize, and
label that evidence as DXGI acceptance rather than display delivery.
Compare reported SDR reference white and derived headroom with system values.
Verify DXGI device removal, session loss and stale-output errors arrive as
the matching Core errors during native surface operations.
Verify reconfiguration rejects acquired images and unretired attachment
or resolve commands, then succeeds after their fence retires even while
another surface has work in flight.

**Deferred adaptive FIFO:** DXGI flip presentation with sync interval zero
can interrupt the prior frame and discard queued frames. Reopen once Core
defines the ordering and tearing rules for late frames and a DXGI route
can demonstrate those guarantees with display evidence.

**Mailbox mode (open).** Core already names this mode. Fluxel DX12 admits
FIFO and, when tearing is supported, immediate presentation. The reference
DX12 backend at wgpu revision `e801fa21` advertises mailbox and uses
`Present(0, 0)`. Specify the observable replacement and tearing guarantees
before enabling this mode. Check support, configuration, and presentation
together on a real window with queued, identifiable frames. Keep mailbox
closed on routes where those guarantees cannot be demonstrated.

**Bounded surface acquisition (deferred).** Fluxel DX12 obtains the current
backbuffer immediately and allows one outstanding image lease. Configuring
maximum frame latency does not provide a caller-visible readiness result.
Reopen when a Host or Renderer caller needs to poll or wait for a bounded
interval before acquiring. Qualify zero timeout, expiry without acquiring
a lease, subsequent success, resize and device loss. The caller chooses
when to wait.

### DX12 upload, completion, and texture transfer verification

**Status:** Direct staging writes, bounded completion waits, sampled color
component mapping, and descriptor-specific sample count reporting are wired.
The implementation has passed a Windows MSVC workspace development build.
Focused contract and GPU evidence are pending the implementation-stage report.
Readback tickets consume explicit submission and completion evidence. They do
not establish a global per-resource hazard tracker, insert implicit waits, or
make ordinary default-heap buffers directly mappable.

**Acceptance:** Read back a direct staging write and verify canceled writes
leave destination bytes and later command recording intact. Verify that a
direct texture staging writer can fill an array or volume using the reported
image and row strides, and that cancellation preserves the texture and later
recording. Cover mixed buffer and texture writes using a nonzero aligned
upload-page offset, and reject depth uploads on the compute lane. Verify that a
pending completion times out and remains positive completion evidence only
after the native fence reaches it. Read back identity, rearranged, and constant
sampled components, and reject sampled-only views as storage or attachments.
Verify that a pending or failed completion prevents readback mapping, and
that a DX12 readback ticket and a direct read lease cannot map the same
buffer concurrently.
Exercise mapping after a caller-supplied completion on a DX12 readback
buffer. Cancel a pending wait before mapping, verify that a failed point
does not open a lease, and remap after explicit release.
Verify a fully initialized device buffer through an encoder upload and a
covering readback. Reject a mismatched encoder, missing copy usage, or data
that does not cover the whole buffer before allocating it.
Verify complete staged texture initialization across multiple mips, including
all array layers and a volume with shrinking depth. Check that malformed
input cannot yield a submittable partially initialized texture. Verify that
an imported texture retains its descriptor when exposed as a Core texture.
On an isolated DX12 device, verify native removal wakes a pending completion
with `DeviceLost`. Check that new encoders, idle waits, and submissions reject
the removed device before allocating command objects or issuing queue work.
Inject a terminal signal failure after a sparse-only handoff and verify
that dropping public device and resource handles preserves quarantined
texture and heap owners.
Verify explicit sampled, storage and attachment usage against actual
descriptor binding and attachment admission. Cover a sampled mip chain
paired with single-mip storage views, and separate sampled depth/stencil
aspects paired with a full attachment view.
Verify a nonzero 3D attachment slice at a nonzero mip through drawing,
float clears and integer clears. Read back adjacent slices to check
isolation. Reject volume resolve destinations. Reopen 2D MSAA resolve to
a volume slice only with a specified resolve-and-copy path and workload.
Check the requested sample count and the reported alternatives against texture
creation for the same format, dimension, mip count, and usage.
Compare storage layout admission with actual typed UAV view creation for
matching formats and dimensions, and verify missing view roles fail before
descriptor copies.
Use the exact format query to cover typed load, store and read-write layouts.
On a device with store-only formats, verify that the allocation and write
path succeed and that a read or read-write layout is refused.
Run direct color copies in both directions using texel-aligned offset and
pitch on a device that reports the feature. Check block alignment, array and
volume image strides, the exact final row, and rejection on the legacy path.

### DX12 comparison sampler integration

Scalar comparison samplers use the existing RHI layout and sampler descriptors.
The DX12 bind group checks the native sampler mode against the declared mode.
Rendering integration needs GPU evidence for a sampled depth view and a
comparison sampler in the same bind group.

**Acceptance:** Read back a depth-comparison result with distinct passing and
failing references, and verify that mismatched sampler modes are rejected.
Exercise fixed and runtime sampler arrays, including partially bound arrays
whose shaders access only initialized entries. Verify mode mismatches are
rejected before descriptor table writes.

### P7-LOWERING-PROFILE — expose cost-bearing lowering facts to RenderGraph

**Ordered usage queries (conditional).** The reference DX12 adapter reports
ordered buffer and texture usages to its tracker. Fluxel already records
explicit barriers. Consider a read-only device fact if RenderGraph can show
a concrete same-queue sequence where existing facts cannot justify omitting
a barrier. Qualify the exact access pair, resource scope and ordering
preconditions on the target backend. RenderGraph decides when to use it.

**Status:** Core exposes an optional per-device profile for GPU-count draws.
DX12 publishes its direct native route and its two vertex-index adjustment
routes. Route tests and classifications for the remaining backends are still
pending. WebGPU currently preprocesses GPU arguments, may allocate temporary
buffers, splits the pass, and emits repeated native draw calls. The P7 store-op
guard prevents one incorrect combination.

**Reopen/selection:** Keep scope limited to capabilities where a backend
lowering changes graph correctness, resource planning, synchronization, or
material CPU/GPU cost. Do not add a general-purpose cost model for all API
operations.

**Acceptance:** Define an immutable per-device lowering descriptor for the
selected cost-bearing operation(s), classify every implemented backend route
from source, and expose facts needed before graph compilation without adding
per-draw queries or allocation. Demonstrate that a caller can distinguish
native execution from GPU preprocessing, pass splitting, temporary resources,
and CPU iteration, then verify descriptors against backend route tests.

### P7-WEBGPU-SPLIT-STATE-COST — bound ordinary pass-state tracking overhead

**Status:** `pass_split::record` creates JavaScript objects/arrays and linearly
searches retained state for ordinary pipeline and binding updates, even when
the pass is never split. This is a concrete extra operation on the common path;
no profile currently proves its frame-time impact.

**Acceptance:** Add focused call/allocation counters or a representative CPU
profile that separates ordinary raster recording from split recovery. Only
then choose direct-indexed state or an explicit split-eligible pass contract;
verify unchanged state restoration after multiple splits.

### P7-RG-FRAMES-IN-FLIGHT — exercise cached plans with multiple live frames

**Status:** The example frame helper constructs a small plan per invocation and
waits for device idle after each frame. It is a smoke path and does not measure
cached-plan reuse, completion-frontier retirement, or overlapping frames.

**Acceptance:** Add one bounded RG-like execution workload that reuses a
compiled plan, submits 2–3 frames before waiting, keeps distinct completion
points, retires a recycled resource only after its covering completion, and
records CPU submit and GPU timings. Do not migrate all numbered examples to
this harness.

Inject retirement registration failure on one DX12 lane and verify that
work on the other lane still retires autonomously at its own fence.
Fail a later handoff after a lane signal. Verify earlier owners retire at
that signal and later unsignaled owners stay quarantined, including sparse
mapping owners.

### P4-AS-ARRAY-VK — Vulkan fixed acceleration-structure arrays

**Status:** Unavailable; the bounded reconstruction failed and temporary
capability/admission changes were withdrawn. The distinct-TLAS result was
`[miss, hit, miss, miss]`, not the required `[hit, miss, hit, miss]`.

**Reopen when:** A concrete implementation fix, new source evidence, or a
relevant device/driver change is available. Do not restart the same fixture
without a discriminating change.

**Acceptance:** Validate the exact artifact and layout route before native
creation; prove four distinct-element hit/miss results on a real GPU; reject
unsupported shapes before native calls; and prove referenced acceleration
structures remain alive through completion after caller handles are dropped.

### P4-MESH-BATCH — Vulkan enhanced mesh batch/count fixture

**Status:** The enhanced fixture remains incomplete after repeated hangs. The
bounded static review found no source-proven correction. Existing verified
direct/indirect/task mesh routes remain as-is; the added count/sentinel claims
are not supported.

**Reopen when:** New fixture or runtime evidence identifies a concrete cause,
or the relevant target device/runtime changes. Do not repeat an unchanged
hanging run.

**Acceptance:** One bounded fixture must cover zero and boundary counts,
nonzero offsets, sentinel preservation, invalid usage/ranges, GPU-produced
arguments/counts, and in-flight resource retention, with distinguishable
real-GPU output and focused regression checks.

### P6-VIDEO-SAMPLING — host-owned video source integration

**Status:** The opaque `VideoFrame` API and WebGPU external-texture wrapper are
implemented. RHI does not decode video or own source creation, frame
scheduling, color conversion, or source closure. Real-source tests are deferred
to integration with `fluxel-host` as requested.

**Acceptance:** On a real browser GPU, separately import a `VideoFrame` and an
`HTMLVideoElement`; bind each through one fully-bound fragment external-video
entry and compare sampled readback against known frames with a documented
color-conversion tolerance. Drop public frame and bind-group handles after
recording and prove source roots remain alive through queue completion. Verify
that successive imports are distinct snapshots, that closing a `VideoFrame`
requires re-import, and that `HTMLVideoElement` import/encode/submit respects
browser task lifetime. Cover source failure, device loss, and cancellation
without sampling a stale frame. Keep source and decoder ownership in the host
adapter.

## Evidence boundary

Configured Surface lease invalidation at `b5d569f` has Core contract tests,
one real WebGPU canvas-resize refusal/reconfigure test, and Rendering's named
DX12/WGL presentation/resize evidence. Vulkan replacement-creation failure
after old-swapchain retirement and Metal out-of-band drawable fact mismatch
remain source/build evidence only. Reopen those native fault paths when a
reproducible failure-injection harness or Apple execution target is selected;
acceptance must show `SurfaceOutdated` and no old-generation lease publication.
See [the scoped validation record](documents/verification.md#configured-surface-lease-facts-2026-11-04).

Metal currently has no Apple GPU execution or Surface evidence in this
repository. This is not a claim of runtime acceptance; compile and source
review results must remain labeled accordingly. A backend capability claim
requiring GPU or presentation behavior needs real Apple-target evidence.

## Closed scope decisions

- **P7-WEBGPU-SPLIT-STORE:** User-owned writable attachments with `StoreOp::Discard`
  are rejected before internal pass termination. The backend-owned dummy depth
  attachment is exempt and cleared after every split. On commit `5866f9f`, the
  Chrome 156 browser suite passed the split-store cases (5/5) and the GPU-count
  readback/query case (1/1). Supporting logs are retained locally and are not
  part of the published source tree. The source is
  published as `fluxel-rhi-webgpu` 0.1.1; 0.1.0 remains immutable and predates
  this fix. Chrome did not expose physical GPU/driver identity, so closure is
  browser-route correctness evidence, not hardware-specific acceptance.
- Stage 4's selected route set and Stage 5 attachment-only allocation intent
  are closed. Closure does not claim every format, backend, array shape, tuple,
  or a physical memory/performance benefit.
- The fixed, fully-bound compute WGSL DX12 storage-buffer and uniform-buffer
  non-uniform routes are resolved. Runtime/partial arrays, dynamic offsets,
  non-compute stages, and mismatched shader/layout metadata remain unavailable.
- First-pass test-suite consolidation across the RHI crates is complete. The
  inventories and review record are in the local Temp archive; no backend
  capability change was part of that work.

## Deferred families, not active commitments

These are outside the current Core + DX12 delivery ledger. Reconsider them
only when their owner supplies the required semantics and a bounded workload:

- Metal rasterization-rate maps.
- Graphics debugger capture controls. A RenderDoc capture session is a tool
  integration. Reconsider an opt-in backend hook only if a debugging workflow
  cannot use the existing tools or scoped interop.
- Vulkan local attachment reads or GLES framebuffer fetch. They require a
  RenderGraph tile-local access and ordering contract.
- Public shader compiler IR, allocation thresholds, automatic device-loss
  recovery policy, and actual display-timing feedback. Error correlation and
  DXGI HDR10 metadata are tracked in the active ledger.
- A public explicit device-destruction API beyond verified stop-admission and
  drain. Reopen only for a lifecycle need that existing RAII cannot express.
- Media decoding, source scheduling, color policy, and source closure beyond
  the native planar-resource work tracked by R04.
- A dynamic runtime backend dispatcher. Typed static backend selection remains
  the Core model.

### Core and DX12 conflict delivery ledger

The ledger covers 28 decisions, R01–R25 and N01–N03. R24 has five independent
workloads under the same parent decision. Design, implementation, verification
and deferral describe the next work on an item. A completed item must become
`IMPLEMENTED`, `LIMITED`, `UNSUPPORTED`, or `OUT_OF_SCOPE`, with its evidence
and exact rejection boundary recorded here. Each active item requires a Core
contract and ownership review before SPI and DX12 work, then a targeted
contract test and a GPU, presentation, or native admission test as applicable.
Work proceeds one feature group at a time, starting with adapter facts in D1.1.

Compare execution mechanisms primarily against `wgpu-hal` and native D3D12.
An operation used by `wgpu` or implemented in `wgpu-core` can also reveal a
missing primitive. For each candidate, identify the explicit operation,
capability query or observable execution result a caller needs. Check the
existing Core/SPI contract and DX12 route for an equivalent before opening
a gap. A present route lacking target evidence needs verification.
RenderGraph and Renderer choose resource states, dependencies, reuse and
logical retirement. RHI executes the supplied transitions, waits and signals,
retains native objects needed for issued work, and reports completion or
device loss. Frame pacing and recovery belong to Renderer and Host.

- **R01, VERIFYING — pre-device surface compatibility probe.** DX12 now
  borrows a Win32 window through `Instance::probe_surface`. An explicit D3D12
  feature-level rejection yields incompatible. Output enumeration provides an
  affinity hint with unknown compatibility. DXGI validates the live route
  during device-bound surface creation and configuration. Verify missing or
  changed output, invalid and foreign inputs, hybrid adapters, and post-open
  revalidation.
- **R02, VERIFYING — error correlation and terminal device loss.** Verify
  DX12 failure reports identify the plan, operation, step and lane where
  applicable, payloads already handed off, and exact checkpoints for the
  successfully signaled lane prefixes. Cover an unsealed suffix and a
  retirement registration failure after valid signals.
  Automatic renderer reconstruction is `OUT_OF_SCOPE`: Renderer and Host own
  graph rebuild, window lifecycle, and policy after a terminal device loss.
- **R03, VERIFYING — shared stop-admission and drain.** Verify that device
  clones reject new encoders, submissions, map leases, and readback recording
  after removal. Pending unsignaled private readbacks must report `DeviceLost`.
  Verify issued owners remain retained until a valid fence completes or the
  uncertain work is quarantined. Confirm existing leases release normally and
  pending waits report a terminal error. An explicit
  public close operation remains `OUT_OF_SCOPE` until a consumer demonstrates
  a lifecycle need that RAII cannot express. Immediate native release is never
  a valid implementation.
- **R04, DESIGNING — native planar textures.** Scope Core/DX12 work to NV12 and
  P010 resource creation, plane views, copy, synchronization, readback, and
  externally owned native-resource import. Decoder ownership, color-policy,
  frame scheduling, and media-source lifecycle are `OUT_OF_SCOPE` to Host.
  Acceptance requires plane-specific positive and rejection cases with GPU
  readback.
- **R05, UNSUPPORTED — direct MSAA resolve to a 3D slice.** DX12 resolve cannot
  represent that pair of shapes. A separately ordered 2D resolve followed by
  copy or compute may be introduced only with its own states, retention, cost,
  and GPU evidence.
- **R06, OUT_OF_SCOPE — public Naga IR.** Compiler IR remains private. It would
  create a versioned public compiler contract without an RHI consumer. Reopen
  only for a demonstrated cross-backend compiler-tool contract.
- **R07, DESIGNING — explicit native binding metadata.** Retain inference
  only for unambiguous binding shapes. Define explicit register/view/immediate
  metadata for HLSL/DXIL shapes reflection cannot distinguish, and test source,
  DXIL, and conflicting-register admission separately.
- **R08, VERIFYING — WGSL subgroup identity, count, and barrier.** Keep only
  operations with exact native wave semantics. Invocation index arithmetic and
  collectives do not derive wave identity/count or memory ordering. Each new
  operation needs a standardized language route, a matching DX12 operation,
  and readback evidence.
- **R09, DESIGNING — WGSL F16 ABI in uniforms and immediates.** Evaluate
  byte-addressed lowering for legal two-byte offsets and nested layouts. Accept
  a shape only after offset, update, and readback tests. Root constants retain
  their DWORD budget and alignment constraints.
- **R10, UNSUPPORTED — WGSL I16/U16.** Fluxel accepts standard WGSL and
  standardized WGSL extensions. It will not import a backend-specific shader
  language extension for these types. Native HLSL/DXIL 16-bit integers remain
  separately feature-gated. Reopen only after a standardized WGSL extension
  defines arithmetic, layout, validation, and target availability.
- **R11, UNSUPPORTED — WGSL 64-bit scalar routes.** Standard WGSL currently
  supplies no 64-bit integer or floating scalar route. Native HLSL/DXIL
  capability remains separately gated and does not extend WGSL. Reopen only
  after a standardized scalar route, frontend support, exact lowering, and
  boundary-value evidence exist. Do not add private WGSL syntax.
- **R12, DESIGNING — standard WGSL immediates.** Validate the source type
  before DX12 lowering. Standard WGSL excludes arrays, including arrays
  inside nested structures. For eligible matrices and array-free structures,
  verify byte offsets, indexing, partial writes, stage ranges and the
  root-signature budget. Track any array experiment under its native or
  compatibility input route. See [inline constants](#dx12-wgsl-inline-constants)
  and the [WGSL variable declaration rules](https://www.w3.org/TR/2026/CRD-WGSL-20261106/#var-decls).
- **R13, DEFERRED — unrestricted WGSL pointer parameters.** The blocking
  condition is compiler/frontend and lowering support, rather than absence of
  a language extension. Current routes cannot preserve general address-space,
  aliasing, and offset semantics. Existing supported composite access remains
  available. Reopen with compiler validation and lowering tests.
- **R14, VERIFYING — primitive index in mesh pipelines.** Raster primitive index
  remains supported. Mesh output needs a separately named, mesh-produced
  semantic if a consumer requires it, because it is not the same value.
- **R15, VERIFYING — conservative depth.** Expose only modes whose early-depth,
  discard, depth-write, and side-effect observations are preserved. Force and
  Allow modes remain separate admission paths.
- **R16, UNSUPPORTED — triangle point fill.** Point-list topology is not an
  equivalent rendering mode. Reopen only for an explicitly specified lowering
  with coverage, reuse, and cost evidence.
- **R17, VERIFYING — query statistics.** Keep `QueryKind::PipelineStatistics` as
  the native interval query and preserve the existing pass-statistics contract
  where its fields are available. Do not advertise a DX12 boundary snapshot
  derived from interval totals. Acceptance covers query interval/readback and
  every claimed pass-statistics field.
- **R18, DESIGNING — indexed BLAS refit.** Define an explicit immutable-input
  contract or a backend-owned snapshot used by both the initial build and every
  update. Buffer identity and count alone are insufficient. Caller/Renderer
  owns scheduling and barriers. Acceptance requires the chosen input-mode
  checks, lifetime retention, refit output, and measured snapshot cost.
- **R19, DESIGNING — GPU-generated TLAS instances.** Use logical GPU instance
  records and a bounded typed BLAS table. DX12 packs private native records and
  retains all referenced acceleration structures. Define invalid-index behavior.
  Caller/Renderer supplies producer ordering and completion coverage. Prove
  typed conversion and retention with GPU tests without implicit synchronization.
- **R20, UNSUPPORTED — mesh point output.** The DX12 mesh route exposes line
  and triangle output. Reopen only with native support or a separately defined
  semantic and observable workload.
- **R21, UNSUPPORTED — FIFO-relaxed present mode.** DXGI interval zero may drop
  queued frames and cannot prove FIFO order. Reopen only with a route that
  proves ordering, late-frame behavior, and the tearing policy.
- **R22, DESIGNING — opt-in HDR10 metadata.** Add a narrow write/clear path
  after defining its color-space, output-selection, and resize lifetime
  contract. Its result means only that DXGI accepted the request. Actual
  scanout timing is `OUT_OF_SCOPE` until a trustworthy timing source and
  visible-output evidence exist. Present receipts and metadata acceptance do
  not prove display delivery.
- **R23, DESIGNING — damage rectangles.** Explore an opt-in preserved-content
  surface route. Flip-discard cannot support it. Renderer owns prior-frame
  history and redraw decisions. The RHI does not insert hidden copies or
  synchronization. Acceptance requires pixels to remain correct across
  back-buffer rotation, resize, and surface generation.
- **R24 — optional DX12 execution mechanisms.** Qualify each of the five
  shapes below independently. A bounded GPU workload can exercise a native
  mechanism. A shared Core operation also needs a meaningful caller contract.
- **R24a, DESIGNING — variable-rate shading.** Define feature tiers, rate
  sources, combiners, resource identity, and pass compatibility before DX12
  lowering. Acceptance requires a capability rejection test and distinguishable
  GPU output or counters.
- **R24b, DESIGNING — sampler feedback.** Define feedback-map resources,
  resolve, synchronization, and tier restrictions. Acceptance requires GPU
  feedback readback. Capability reporting alone is insufficient.
- **R24c, DESIGNING — geometry and tessellation stages.** Define shader-stage
  linkage, reflection, pipeline ABI, limits, and DXIL/HLSL/WGSL route facts.
  Verify geometry-shader amplification and hull/domain patch evaluation with
  separate outputs. Microsoft's PC DX12 Bézier sample supplies a tessellation
  reference. [SimpleBezierPC12](https://github.com/microsoft/Xbox-ATG-Samples/tree/ff629a6e4876e38fe8761ead116a0e53f634ca67/PCSamples/IntroGraphics/SimpleBezierPC12)
- **R24d, DESIGNING — programmable sample positions.** Define tier gate,
  sample-position descriptors, pipeline/pass compatibility, and lifetime.
  Test a repeated one-pixel pattern on tier 1 and a four-pixel pattern on
  tier 2. Read back known edge coverage. Exercise depth resolve and full
  depth/stencil copy with the sample pattern used to write the source.
  The caller supplies the pattern and resource dependencies.
  [Programmable sample positions](https://github.com/microsoft/DirectX-Specs/blob/ae2cc81d8f97df9ce8597e25d84405273f09d08c/d3d/ProgrammableSamplePositions.md)
- **R24e, DESIGNING — predication.** Define condition-buffer format, command
  range, queue ordering, and retained ownership. Acceptance requires true and
  false GPU cases with distinct output.
- **R25, VERIFYING — native interop.** Complete narrowly typed interop through
  existing static backend boundaries where a real consumer needs it. A dynamic
  backend dispatcher is `OUT_OF_SCOPE`: it would weaken typed device identity
  and introduce runtime dispatch into the public RHI model.
- **N01, VERIFYING — CPU-visible DX12 buffers.** Explicit standard upload
  and readback placement and the OPTIONS16-gated GPU upload heap route are
  implemented. Verify the actual heap type, CPU write lease, direct GPU reads
  and readback, and early rejection for unsupported devices, GPU write usage,
  and read mapping. The GPU upload route keeps the native generic-read state.
  Shared-handle import and export of GPU upload heaps are unsupported because
  D3D12 prohibits shared flags for this heap type.
  Verify custom placement against the node's equivalent upload, readback and
  GPU upload heap profiles, including a coherent UMA write-back upload.
  Reject cache and pool requests outside those profiles. In particular, a
  discrete device-local readback heap has no supported native counterpart.
  UMA topology alone does not make a default-heap buffer mappable.
- **N02, VERIFYING — safe completion-bound mapping.** DX12 private staging
  tickets record the copied range and the first covering backend signal on the
  issuing lane. Verify pending status before submission and before fence
  completion, exact readback bytes, discarded recordings, rejected packets,
  and covered versus uncertain work after a later handoff failure. Confirm
  that an unrelated completion cannot authorize mapping a pending private
  ticket. Shared readback buffers retain explicit caller-provided coverage
  and exclusion from later writes because the ticket cannot own those hazards.
- **N03, VERIFYING — retryable DX12 partial submission.** DX12 returns the
  unissued suffix when a step fails its clean boundary check for a stale
  present lease or unavailable lane fence values. Check emergency completion
  on every accepted lane with `CompletionRequest::None`, rejection of missing,
  wrong-lane, and pending frontier points, and suffix-only resume with
  cross-lane dependencies on the completed prefix. Check that uncertain native
  wait, execute, signal and present handoffs never produce a retryable suffix.
  Verify one-shot external waits remain unconsumed when the step stops before
  handoff and are returned in the multi-queue diagnostic tail.

Renderer algorithms, frame pacing, host and decoder lifecycle, cache eviction,
and asset residency remain owned above RHI.
