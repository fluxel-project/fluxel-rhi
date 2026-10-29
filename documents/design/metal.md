# Metal Backend Design

`crates/metal` lowers Core operations to Apple Metal. Native objects remain
private to the crate. The backend reports native execution and completion
facts; RenderGraph remains responsible for work ordering, hazard planning,
frame policy, and resource retirement.

## Device, command, and completion ownership

Device state owns the selected Metal device, command queues, capability facts,
submission tracking, bounded upload state, and native helper caches. The
ordinary execution path records a one-shot native command buffer in Core order
and submits it without building a CPU replay list.

Completion handlers and joins describe native completion. Cancelling an
observer does not cancel work already committed to Metal. A drawable from a
`CAMetalLayer` is a generation-bound lease that can be handed off once for
presentation. The host owns the layer and its lifecycle.

## Synchronization and resource transitions

Tracked resource operations lower caller-supplied execution intent. Placement
heaps and other untracked or alias transitions use a narrow command-buffer
event boundary where required; this is not a substitute for general resource
state tracking or hazard inference.

Metal can expose an optional second queue and shared-event waits only where the
native path supports the explicit contract. Queue and packet restrictions stay
visible; the backend does not imply arbitrary cross-packet dependencies.

## Shader and binding ABI

WGSL lowers through Naga to MSL. Direct MSL and precompiled library routes are
separate explicit inputs. CPU-only translation can use bounded worker
execution, while Objective-C object creation, native callbacks, resource
creation, and command encoding remain on their native ownership path.

The scalar binding ABI maps each Core binding to a fixed native slot:
`7 * group + binding`. Groups occupy the supported buffer-slot range while
vertex buffers use the reserved upper slots. The backend fails closed when a
layout exceeds this ABI rather than silently assuming an unimplemented
argument-buffer layout. Argument-buffer array materialization uses fresh
storage for each logical element so later binding updates cannot mutate data
already consumed by the GPU.

## Supported scope and limits

Ray acceleration-structure construction does not imply ray-pipeline tracing.
Binding arrays, mesh batching, external-video import, and native operation
shapes remain individually capability-gated. PSO archives do not establish a
DXR-style state-object cache. Unsupported routes return structured errors
rather than being simulated on the CPU.

No Apple GPU execution or presentation evidence is claimed by the current
repository record. Cross-compilation or static review cannot substitute for
Apple-target GPU and Surface acceptance when a capability claim requires it.
The old archive-prepare timing reporter had no correctness oracle; its latency
remains unmeasured and should only be revisited with a representative
benchmark workload.
