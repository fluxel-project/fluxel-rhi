# `fluxel-rhi-dx12`

`fluxel-rhi-dx12` implements the Fluxel RHI backend for Direct3D 12. It lowers
the shared Core contract to D3D12 resources, command lists, descriptors,
fences, and DXGI presentation.

## Platform and requirements

- Windows only, using the MSVC toolchain and a Direct3D 12 capable adapter.
- DXGI and D3D12 runtime support are required. DXC is required for WGSL/HLSL
  shader compilation routes.
- The default `dx12` feature of `fluxel-rhi` selects this backend on Windows.
  Applications may disable facade defaults and enable `dx12` explicitly.

## Use through the facade

```toml
[target.'cfg(windows)'.dependencies]
fluxel-rhi = { version = "0.16.0", default-features = false, features = ["dx12"] }
```

In an async application setup routine:

```rust
let instance = fluxel_rhi::dx12::Dx12Instance::new()?;
let device = instance.open_default().await?;
```

Surface creation borrows a live host-owned Win32 window handle and is unsafe:
the window and its handles must outlive the surface and all image leases. Use
the host integration guide before creating a surface.

## Capabilities and design

Capabilities depend on the selected adapter, descriptor, shader route, and
native tier. Query the device and reject unsupported routes explicitly. See
[`DESIGN.md`](DESIGN.md) for resource, submission, and capability boundaries;
the [examples guide](../rhi/examples/README.md) for runnable workloads; and the
[verification guide](https://github.com/fluxel-project/fluxel-rhi/blob/main/documents/verification.md)
for hardware-test requirements.

The `hardware-tests` feature enables opt-in real-device fixtures. It is intended
for backend development and is not needed by library consumers.

Requires Rust 1.87 or newer. Licensed under `MIT OR Apache-2.0`.
