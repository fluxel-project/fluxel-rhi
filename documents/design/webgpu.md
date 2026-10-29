# WebGPU Backend Design

`crates/webgpu` implements the Core contract through browser WebGPU on
`wasm32`. Browser objects and JavaScript ownership stay inside the backend and
its owner thread. Core receives typed resources and execution facts, not DOM,
`GPUDevice`, canvas, or JavaScript session objects.

## Browser ownership and command flow

The device and browser resources are owner-thread local. RHI commands encode
directly into `GPUCommandEncoder` and native pass encoders; they are not stored
as a portable CPU command stream. Submission finishes the native command
buffer and sends it to the WebGPU queue.

The encoder retains JavaScript roots required by the submitted work. Queue
completion settles a submitted prefix and permits those roots to be released.
Dropping a Rust completion observer cancels observation, not native work. On
device loss, completion fails while the backend preserves native dependencies
until the browser establishes their terminal disposition.

## Surface and shader architecture

The backend configures the WebGPU canvas and manages the generation-bound
current-texture lease. DOM sizing, canvas lifecycle, frame scheduling, and
device restoration belong to the host adapter. A resize or reconfiguration
invalidates stale surface images.

WGSL is the native shader language. SPIR-V translation and validation run in a
browser worker that owns no WebGPU objects; GPU resources and command encoding
remain on the browser owner thread. Upload alignment and unsupported copy
shapes use retained staging pages or GPU-side repacking rather than hidden CPU
readback.

## Capabilities and video boundary

Optional standard features are requested and reported only when exposed by the
adapter. Format, usage, binding, and operation checks remain specific to the
native route. GPU-counted indirect execution transforms arguments on the GPU;
it does not read the count on the CPU or replay application commands.

`VideoFrame` import is a separate external-video binding domain. It wraps an
opaque source and keeps the JavaScript source root alive through queue
completion. The backend does not decode video, expose plane formats, convert
color, schedule frames, or own source creation and closure. Other backends do
not claim this route. Real browser-source integration remains open under
[`todo.md`](todo.md).

Standard WebGPU does not provide routes for every Core domain. In particular,
ray pipelines, mesh shaders, sparse binding, independent native queues,
placement heaps, native secondary recording, full pipeline statistics, and
some presentation/copy shapes remain unavailable unless a concrete WebGPU
mechanism and validation support them. Unsupported operations fail explicitly.
