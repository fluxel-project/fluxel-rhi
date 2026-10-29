# `fluxel-rhi`

`fluxel-rhi` is the application-facing facade for Fluxel's statically typed GPU
execution API. It reexports Core and exposes selected backend crates through
Cargo features. It does not choose a runtime backend enum: select a concrete
backend when constructing the device.

## Choose a backend

```toml
[dependencies]
fluxel-rhi = { version = "0.1.0", default-features = false, features = ["vulkan"] }
```

Available backend features are `dx12` (Windows), `vulkan`, `webgpu` (`wasm32`),
`native-gl-wgl` (Windows), `native-gles-egl`, `webgl2` (`wasm32`), and `metal`
(Apple targets). The facade defaults to `dx12`; disable defaults when selecting
another backend explicitly.

```rust
use fluxel_rhi::api;
use fluxel_rhi::vulkan::VulkanInstance;

// In an async application setup routine:
let instance = VulkanInstance::new()?;
let device = instance.open_default().await?;
```

The caller or host adapter owns window/canvas lifecycle and supplies the
appropriate surface integration. Device capabilities depend on the selected
adapter and requested route.

## Examples and documentation

The `examples` feature enables 23 numbered workloads. Start with
[`01_triangle`](examples/01_triangle/main.rs); the [examples guide](examples/README.md)
documents backend selection, browser/Android setup, required assets, and
feature fallbacks. Some examples require local data; see
[the asset guide](examples/assets/README.md).

The [workspace design overview](DESIGN.md) explains ownership boundaries. The
[verification guide](https://github.com/fluxel-project/fluxel-rhi/blob/main/documents/verification.md)
covers tests, hardware evidence, and platform-specific setup.

Requires Rust 1.87 or newer; licensed under `MIT OR Apache-2.0`.

Repository: https://github.com/fluxel-project/fluxel-rhi
