# DX12 test layout

DX12 keeps private backend assertions in `src/tests/`; integration GPU cases live in `tests/` and use the public RHI surface. Shader-lowering tests are children of the owning shader backend module, so a test run does not recompile that backend once per integration target.

Root integration files remain separate Cargo targets when their process-local diagnostic counters or environment setup should stay isolated. Repeated synchronous completion polling is shared by `support/mod.rs`. Nested target modules collect related cases while preserving their individual test names:

- `clear_gpu/`: invalid clear descriptions, depth tests, and color subresource selection.
- `mesh_gpu/`: MRT/stencil, depth controls, multi-indirect/count, and descriptor reservation release.
- `sparse_gpu/`: tile mapping/remapping and packed mip tails.
- `mrt_gpu/` and `readonly_depth_stencil_gpu/`: raster attachment suites.
- `assets/`: source and checked DXIL fixtures used by real-device tests.

Texture test targets combine only cases with repeated creation/encoding paths; they retain separate GPU tests for each format, view dimension, shader route, and readback oracle. In particular, depth formats, MSAA dimensions, D1/D2/D3 sampled views, cube views, and BC mip cases remain independently filterable.

Compile every DX12 hardware-test target without running it:

```powershell
cargo test -p fluxel-rhi-dx12 --features hardware-tests --no-run
```

Run one focused GPU target on a DX12 device:

```powershell
cargo test -p fluxel-rhi-dx12 --test sampled_texture_readback_gpu --features hardware-tests -- --test-threads=1
```
