# RHI testing

## What is present

This checkout has an active DX12 backend and the typed facade/examples. It does not
contain the historical Vulkan, Metal, or WebGPU backend test suites referenced by older
documentation. Commands for those removed crates are invalid and are not conformance
evidence.

Run the portable API/unit coverage with:

```powershell
cargo test -p fluxel-rhi-core
cargo test -p fluxel-rhi-dx12
```

Check an example through the facade with:

```powershell
cargo check -p fluxel-rhi --features examples --example 10_indirect_draw
```

On a Windows DX12 machine, run a finite example workload when it supports `--frames`:

```powershell
cargo run -p fluxel-rhi --features examples --example 10_indirect_draw -- --frames 2
```

The example command is real GPU evidence for the exercised route only. It is not a
claim of general backend conformance.

## Evidence rules

Record the command, commit, Windows version, adapter/driver, workload and observed
outcome for a hardware claim. Classify an outcome as follows:

- **Pass:** the selected native route ran and its stated result was observed.
- **Unsupported:** the backend accurately rejected a capability or descriptor it does
  not implement.
- **Skipped:** the required platform, adapter, window host or driver was unavailable.
- **Failure:** an advertised route failed, lost the device or produced incorrect output.

Compilation verifies API integration; it does not prove native command execution.
Hardware evidence must name the exact route: for example a D2 array texture view,
indexed indirect draw, a depth attachment, or a readback path.

## Adding coverage

Keep portable behavior tests in `crates/core/src/api/tests.rs` when they exercise the
contract without a device. Put DX12-native validation beside the active DX12 crate and
use the public `api` descriptors wherever possible. Do not revive test paths for absent
backends or private constructors merely to make a matrix cell appear covered.
