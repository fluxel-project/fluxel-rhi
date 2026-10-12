# Fluxel RHI

Fluxel RHI provides a statically typed GPU execution API for Fluxel. RenderGraph and Renderer choose work, synchronization, and resource lifetimes; RHI reports device capabilities, creates native resources, executes commands, presents images, and reports completion.

The workspace currently contains three crates:

| Crate | Responsibility |
| --- | --- |
| `fluxel-rhi-core` | Portable types and backend contracts in `api` |
| `fluxel-rhi-dx12` | Direct3D 12 implementation |
| `fluxel-rhi` | Public facade that reexports `api` and the DX12 backend |

The public contract is split by concern under [`crates/core/src/api`](crates/core/src/api/mod.rs). Backend implementations use the same descriptors and operation meanings as the public API. A native host selects a concrete backend once; resources, encoders, and submissions retain that backend type.

## Build and run

On Windows:

```powershell
cargo check --workspace --all-targets
cargo test -p fluxel-rhi-core --lib
cargo run -p fluxel-rhi --features examples --example 01_triangle -- --frames 2
```

The numbered examples are in [`crates/rhi/examples`](crates/rhi/examples). Each example chooses the DX12 host and uses `fluxel_rhi::api::*` for GPU work. `--frames 2` closes the native window after two frames, which is useful for automated runs. Some examples need local model and texture assets; see the [asset instructions](crates/rhi/examples/assets/README.md).

The [RHI design](documents/design-rhi.md) describes responsibilities and capability rules. DX12 implementation details are divided into focused modules under [`crates/dx12/src/backend`](crates/dx12/src/backend.rs).
