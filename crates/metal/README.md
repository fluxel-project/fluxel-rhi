# `fluxel-rhi-metal`

`fluxel-rhi-metal` implements the Fluxel RHI backend for Apple Metal. It lowers
Core operations to native Metal resources and one-shot command buffers, and
reports native completion and presentation facts.

## Platform and requirements

The native backend is available only on Apple targets and requires the Apple
SDK and Metal-capable device. Surface integration uses a host-owned
`CAMetalLayer`; the RHI borrows the layer and does not own application window
lifecycle.

## Use through the facade

```toml
[target.'cfg(target_vendor = "apple")'.dependencies]
fluxel-rhi = { version = "0.16.0", default-features = false, features = ["metal"] }
```

In an async application setup routine:

```rust
let instance = fluxel_rhi::metal::MetalInstance::new()?;
let device = instance.open_default().await?;
```

WGSL lowers through Naga to MSL. Direct MSL and precompiled Metal library
routes are separate inputs. Optional features remain device-gated; acceleration
structure support does not imply ray-pipeline support.

## Design and verification

See [`DESIGN.md`](DESIGN.md) for support limits and native ownership. The
`hardware-tests` feature enables opt-in Apple-device fixtures. This repository
does not currently claim Apple GPU or Surface acceptance; compile and source
review are not runtime evidence. See the [verification guide](https://github.com/fluxel-project/fluxel-rhi/blob/main/documents/verification.md).

Requires Rust 1.87 or newer; licensed under `MIT OR Apache-2.0`.
