# Vulkan Backend Design

`crates/vulkan` lowers Core operations to Vulkan. The crate owns instance and
device selection, native resources, command buffers, descriptors, memory,
presentation, and completion. It reports only features enabled on the selected
logical device and queues that can execute the requested route.

## Device and resource ownership

`VulkanInstance` owns the loader and instance. Device creation selects a
graphics queue and any compatible optional compute queue, enables a specific
feature set, and records those enabled facts for capability and shader
admission. Physical-device potential alone does not enable a Core capability.

Resource, surface, pipeline, memory, sparse, ray, and submission modules keep
native details local. Retained references keep Vulkan parents alive until child
objects are destroyed. Resource allocation pages are keyed by native memory
type, allocation flags, resource class, and mapping requirements. Dedicated
requirements, explicit placement, sparse binding, external imports, and ray
backing use distinct paths.

## Recording, synchronization, and completion

The backend records native command buffers for the caller's ordered plan.
A submission gate preserves packet order; a queue gate provides Vulkan's
required external queue synchronization. Neither lock sorts graph work or
implements a second scheduler.

Fences track accepted submission lifetime. Timeline semaphores carry explicit
cross-queue dependencies where enabled and supported. Completion observers
retain native work through success, device loss, or uncertainty; dropping an
observer only stops observation. Sparse-bind completion follows its native
binding operation and remains separate from renderer policy.

Caller-supplied transitions and barriers determine resource visibility. A
queue wait does not synthesize memory barriers or infer resource state.

## Raster, shaders, and capabilities

Raster work uses Vulkan dynamic rendering. Viewport and front-face lowering
adapt the native convention to the shared framebuffer contract. Runtime
shader artifacts are SPIR-V; maintained WGSL sources may generate artifacts
offline, but there is no hidden runtime compiler fallback.

Ray tracing, mesh shaders, sparse images, descriptor arrays, subgroup controls,
multiview, selected atomics, cooperative matrices, and advanced indirect
operations are feature-gated. Shader admission checks the selected artifact
route and enabled logical-device facts before native shader creation. A
physical-device extension or a successful artifact validation alone does not
establish Core support.

## Boundaries and open work

Vulkan does not infer resource state from creation flags, choose aliases,
fabricate resident-memory statistics, replay commands on the CPU, or enable
features that were not selected at device creation. Runtime descriptor-array
capacity is a caller-selected operational bound, not a claim about a driver
maximum.

Two Vulkan routes remain open: fixed acceleration-structure descriptor arrays
and the enhanced mesh batch/count fixture. Their current status, evidence, and
reopening conditions are in [`todo.md`](todo.md). The supported cooperative
matrix path remains deliberately narrow: selected row-major layout, stride,
pointer provenance, and tuple combinations only; other layouts and tuples are
not implied.
