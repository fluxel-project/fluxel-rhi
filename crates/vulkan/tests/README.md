# Vulkan integration tests

Each integration test target runs in an independent process. Flat root-level
test files are Cargo-discovered; behavior directories use explicit `[[test]]`
entries in `Cargo.toml` to preserve the target name and process boundary.

Current grouped targets:

- `clear_gpu/`: description validation, depth-bias, rectangle, and subresource
  clear behavior within one target.
- `hardware/`: format and texture-plane hardware cases.
- `int64_atomic_gpu/`: shared Int64 route validation plus separate
  StorageBuffer and Workgroup opcode cases within the original target.
- `mesh_gpu/`: direct/indirect entry cases and the shared full dispatch/readback
  scenario within one target.
- `mrt_gpu/`: MRT, resolve and transient attachment output checks.
- `support/`: shared no-op-Waker polling helper. Thread-Waker/park helpers stay
  local to the tests that depend on their wake behavior.
- `readonly_depth_stencil_gpu/`: read-only depth/stencil state conflicts.
- `sampler_gpu/`: clamp-to-zero, sampler-layout retention, runtime mixed
  sampler arrays, and WGSL fixed/runtime sampler arrays. These retain distinct
  shader routes and output oracles.
- `toy/`: shared setup and recording pieces used by the toy scene target.

Build all Vulkan hardware targets without executing the GPU suite:

```powershell
cargo test -p fluxel-rhi-vulkan --features hardware-tests --no-run
```
