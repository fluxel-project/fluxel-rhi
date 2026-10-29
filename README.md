# Fluxel RHI

Fluxel RHI is Fluxel's GPU execution layer. It creates native resources,
executes caller-ordered GPU work, manages presentation leases, and reports
submission and completion facts.

It is designed first for RenderGraph and rendering-pipeline implementations:
a thin, low-cost execution layer that unifies operation semantics while
preserving native GPU capabilities and exposing execution facts that affect
graph planning. Application-level developer convenience is secondary to
predictable execution of already planned work.

RenderGraph and Renderer own dependency planning, hazard analysis, frame policy,
and resource retirement. RHI executes the plan and supplies native mechanisms;
it does not rebuild the graph or choose resource lifetimes.

## Workspace

- **`fluxel-rhi-core`** — portable public API, resource identity, capability
  contracts, and backend SPI.
- **`fluxel-rhi`** — facade that reexports Core and feature-selected backends;
  also owns the numbered integration examples.
- **Backend crates** — `fluxel-rhi-dx12`, `fluxel-rhi-vulkan`,
  `fluxel-rhi-webgpu`, `fluxel-rhi-gl`, and `fluxel-rhi-metal` implement native
  resource management, command lowering, synchronization, and presentation.

Backends are selected through Cargo features at the integration boundary.
Their API types remain statically dispatched; capabilities are reported for
the selected device and requested operation.

## Backend selection

- **DX12** — `dx12` (facade default; Windows target only).
- **Vulkan** — `vulkan`.
- **Browser WebGPU** — `webgpu` on `wasm32`.
- **Desktop GL / WGL** — `native-gl-wgl`.
- **GLES / EGL** — `native-gles-egl`.
- **Browser WebGL2** — `webgl2`.
- **Metal** — `metal` on Apple targets.

See the backend READMEs for setup and the separate `DESIGN.md` files for each
route's architecture and support limits. Cross-backend unfinished work is
tracked in [`TODO.md`](TODO.md).

## Build and test

Run from the workspace root with Rust 1.87 or newer and the toolchain for your
target platform. On Windows, use the MSVC toolchain.

```powershell
cargo check --workspace
cargo test -p fluxel-rhi-core
```

The Core command runs portable Core tests. Backend, browser, and hardware test
commands require their matching target, features, drivers, or browser setup; use
the [verification guide](documents/verification.md) for those commands and for
the evidence each result supports.

## Run examples

Examples use a finite frame count so native smoke runs exit predictably. On
Windows, run DX12 or Vulkan from the workspace root:

```powershell
cargo run -p fluxel-rhi --features examples,dx12 --example 01_triangle -- --frames 2
cargo run -p fluxel-rhi --no-default-features --features examples,vulkan --example 01_triangle -- --backend vulkan --frames 2
```

Native examples and GPU fixtures use the separate [`fluxel-host` package](https://github.com/fluxel-project/fluxel-host) for window and lifecycle events; the RHI crates do not depend on it at runtime.

Some scenes need local model and texture assets. Fetch them before running
those examples:

```powershell
.\tools\examples\fetch-assets.ps1
```

The [examples guide](crates/rhi/examples/README.md) lists all 23 workloads and
covers native, browser, and Android routes. Asset locations and licensing notes
are in the [examples asset guide](crates/rhi/examples/assets/README.md).

## Documentation

- [RHI architecture overview](crates/rhi/DESIGN.md) — ownership and shared
  architecture.
- [Core package](crates/core/README.md) and [Core design](crates/core/DESIGN.md)
  — public contract and backend SPI.
- Backend packages: [DX12](crates/dx12/README.md), [Vulkan](crates/vulkan/README.md),
  [WebGPU](crates/webgpu/README.md), [GL](crates/gl/README.md), and
  [Metal](crates/metal/README.md).
- [Open RHI TODOs](TODO.md) — cross-backend follow-up and evidence gates.
- [Verification guide](documents/verification.md) — tests, examples, hardware
  coverage, and evidence limits.
- [Architecture decision records](documents/adr/README.md) — decision history,
  rationale, and status.
- [Publishing guide](PUBLISHING.md) — package order, packaging checks, and
  known registry blockers.
