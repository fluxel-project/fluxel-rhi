# `fluxel-rhi-vulkan`

`fluxel-rhi-vulkan` implements the Fluxel RHI backend for Vulkan. It owns
Vulkan instance/device selection, native resources, command buffers, memory,
presentation, and completion tracking while preserving the shared Core API.

## Platform and requirements

The crate loads the Vulkan loader at runtime and requests Vulkan 1.3. A
compatible loader, driver, and graphics-capable device are required. Window
presentation also requires a supported raw window/display handle pair and the
corresponding platform surface extensions.

## Use through the facade

```toml
[dependencies]
fluxel-rhi = { version = "0.16.0", default-features = false, features = ["vulkan"] }
```

In an async application setup routine:

```rust
let instance = fluxel_rhi::vulkan::VulkanInstance::new()?;
let device = instance.open_default().await?;
```

Native surface creation borrows platform window/display handles. Keep both
valid until the surface and every image lease have been dropped. Device
capabilities reflect features enabled on the selected logical device, not
everything supported by its physical device.

## Capabilities and design

Optional routes depend on enabled device features, extensions, queue families,
formats, and resource usage. External-memory imports have platform-specific
handle requirements. See [`DESIGN.md`](DESIGN.md) for backend ownership and
support limits, the [examples guide](../rhi/examples/README.md) for workloads,
and the [verification guide](https://github.com/fluxel-project/fluxel-rhi/blob/main/documents/verification.md)
for current hardware evidence.

The `hardware-tests` feature enables opt-in real-device fixtures. It is not
required by library users. Requires Rust 1.87 or newer; licensed under
`MIT OR Apache-2.0`.
