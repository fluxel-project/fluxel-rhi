# Metal implementation review — 2026-10-22

This packet describes implementation and compilation evidence. At the user's
request, this Metal phase does not run tests, examples, or GPU acceptance.
No Apple GPU, presentation, visual, generated-MSL compiler, or performance
result is claimed. Rust compilation does not prove those results.

## Ownership and native execution

Metal resources, heaps, libraries, pipelines, queues, events, argument buffers,
and command buffers remain backend-owned. Calls record native commands directly.
RenderGraph continues to own scheduling, hazards, and retirement policy. Resource
retention, binary archives, and native pipeline caches do not replay CPU commands.

Explicit barriers use native command-buffer events between passes. Cross-queue
work uses shared events and keeps producer completion failures observable.
Sparse placement uses the Metal 4 mapping queue when its runtime capability is
available; a host completion callback hands the native mapping to that queue.
This bridge does not imply CPU-free asynchronous sparse submission.

## Native contract boundaries

| Contract | Boundary |
| --- | --- |
| Ray pipeline | Excluded from this phase by the user because the Metal Shader Converter SDK is unavailable. Metal itself can implement converted ray pipelines; this is not a hardware impossibility. |
| Secondary raster work | Core pre-records viewport/scissor and complete draw state independently of a parent pass. Native ICB render commands have no viewport/scissor commands. Parallel render encoders require the actual parent pass. The backend does not substitute CPU command replay. |
| Eleven-field pipeline statistics | Metal pass counters have different fields and sampling semantics. The separate pass-boundary counter API does not advertise the eleven-field contract. |
| Sparse placement | Requires runtime `supportsPlacementSparse` and Metal 4 queue mapping; Metal 3 automatic sparse heap allocation does not express Core's explicit allocation offsets. |
| Raster modes | Metal does not expose conservative rasterization, point polygon fill, or a fixed pipeline sample mask. All active sample bits must remain enabled. |
| Resource and shader limits | GPU family, selectors, native formats, sample counts, argument-buffer tier and combined per-stage resource slots govern support. Limits are checked for the actual device and pipeline. |
| External handles | Native buffers/textures belong to the target device. A Metal shared texture handle is reopened on its originating device, including another process; it is not a generic cross-GPU handle. |

## Source references

- [Apple Metal feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [Native indirect render commands](https://developer.apple.com/documentation/metal/mtlindirectrendercommand)
- [Parallel render encoders](https://developer.apple.com/documentation/metal/mtlparallelrendercommandencoder)
- [Command-buffer event signals](https://developer.apple.com/documentation/metal/mtlcommandbuffer/encodesignalevent(_:value:))
- [Metal Shader Converter](https://developer.apple.com/metal/shader-converter/)

Implementation and final compile/review completion are recorded separately after
integration. This document alone is not an acceptance claim.
