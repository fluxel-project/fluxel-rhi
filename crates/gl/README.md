# `fluxel-rhi-gl`

`fluxel-rhi-gl` provides one statically dispatched RHI backend for desktop
OpenGL, OpenGL ES, and browser WebGL2. Its profile and capabilities depend on
the selected Cargo feature and the actual context.

Fluxel RHI is designed first for RenderGraph and rendering-pipeline
implementations. This backend keeps its immediate execution route thin and
low-cost, preserves profile-native capabilities, and reports execution facts
needed for graph planning; application-level convenience is secondary.

## Select a profile

- `native-gl-wgl` — Windows desktop GL through WGL.
- `native-gles-egl` — native GLES through EGL.
- `webgl2` — browser WebGL2 on `wasm32`.

There is no default GL provider. Enable the feature matching the target and
context; a feature does not make another profile available.

## Context and thread ownership

GL calls execute on the context owner thread as they are encoded. Keep the
host window/display handles and native context alive for the backend, device,
surface, and resources. The WGL/EGL context-adoption constructors are unsafe
because their handle and thread lifetime requirements are part of correctness.
WebGL2 context loss ends the current generation; create a new instance and
device after restoration.

For applications using the facade, enable the corresponding feature on
`fluxel-rhi`. The GL backend has no independent queue, secondary recording, or
automatic profile fallback. Feature coverage is checked against the selected
context. See [`DESIGN.md`](DESIGN.md), the [examples guide](../rhi/examples/README.md),
and the [verification guide](https://github.com/fluxel-project/fluxel-rhi/blob/main/documents/verification.md).

Requires Rust 1.87 or newer; licensed under `MIT OR Apache-2.0`.

Repository: https://github.com/fluxel-project/fluxel-rhi
