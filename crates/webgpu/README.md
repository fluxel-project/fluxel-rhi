# `fluxel-rhi-webgpu`

`fluxel-rhi-webgpu` implements the Fluxel RHI backend for browser WebGPU. It
uses browser-owned GPU objects, encodes directly into native WebGPU command
encoders, and reports queue completion through the shared Core API.

## Platform and requirements

This crate exposes its backend only on `wasm32` and requires a browser with
WebGPU enabled. `WebGpuInstance` and its resources are confined to the browser
owner thread; they are not a cross-thread GPU device abstraction. The host
owns the canvas and its lifecycle.

## Use through the facade

```toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
fluxel-rhi = { version = "0.16.0", default-features = false, features = ["webgpu"] }
```

In an async browser entry point:

```rust
let instance = fluxel_rhi::webgpu::WebGpuInstance::new()?;
let device = instance.open_default().await?;
let surface = fluxel_rhi::webgpu::WebGpu::surface_from_canvas(&device, canvas)?;
```

WGSL is the native shader route. SPIR-V translation is optional and requires
deploying the generated browser worker glue and adjacent Wasm asset, then
calling `configure_spirv_worker` before creating SPIR-V shaders. The
`WebGpuVideoSource` wrapper borrows an opaque host-created source; it does not
decode video or manage source lifetime.

## Capabilities and design

Standard WebGPU does not expose every Core operation. Unsupported domains are
reported explicitly. See [`DESIGN.md`](DESIGN.md), the [examples guide](../rhi/examples/README.md),
and the [verification guide](https://github.com/fluxel-project/fluxel-rhi/blob/main/documents/verification.md)
for architecture, integration setup, and evidence limits.

Requires Rust 1.87 or newer; licensed under `MIT OR Apache-2.0`.
