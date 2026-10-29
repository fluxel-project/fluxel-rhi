# GL-family regression tests

Tests are owned by the boundary they verify. The GPU suite remains part of the
library test target because several cases deliberately inspect private provider,
query fallback, and native transfer seams. It does not add a public test API.

## Layout

- `backend/`: private ABI, capabilities, shader/lowering, cache and lifecycle checks.
- `gpu/bindings/`: fixed, indexed, partial, runtime and direct-artifact bindings.
- `gpu/textures/`: format, compression, readback, transfer, depth/stencil and MSAA.
- `gpu/raster/`, `gpu/surfaces/`, `gpu/queries/`, `gpu/execution/`: observable GPU behavior.
- `gpu/shaders/artifacts/`: direct shader artifact and native ABI regressions.
- `gpu/platform/`: behavior specific to GLES or WebGL.
- `gpu/diagnostics/`: exhaustive precision coverage, retained with its existing test attributes.
- `gpu/support/`: shared context setup, wake-driven executor, one-step recording
  helper and exact canonical pipeline states. Non-default state remains explicit.

`gpu/mod.rs` is the registration entry. Its module aliases retain cross-fixture
references while source paths describe the behavior under test. Each hardware
case keeps its WGL, GLES or WebGL entry and its original profile conditions.

## Running selected tests

Portable private checks:

```text
cargo test -p fluxel-rhi-gl --lib
```

List native entries or run a selected hardware regression on an owner thread:

```text
cargo test -p fluxel-rhi-gl --features native-gl-wgl --lib -- --list
cargo test -p fluxel-rhi-gl --features native-gl-wgl --lib wgl_core_pbo_upload -- --ignored --test-threads=1
```

GLES uses `native-gles-egl`. Browser tests use the `wasm32-unknown-unknown` target,
`webgl2`, wasm-bindgen-test-runner and a configured real browser/WebDriver.
Compilation alone is not evidence of native or browser GPU correctness.

## Consolidated and removed coverage

- **`core_webgl_pbo_readback`:** Retained in
  `gpu/textures/readback/pbo_readback.rs`, including D2/D3 and RGBA8/sRGB cases.
- **GLES and WebGL UNORM readback helpers:** Retained in
  `gpu/textures/readback/unorm_readback.rs`, preserving each platform's format
  set and whole-buffer padding oracle.
- **ETC2 and BC1 PBO fixtures:** Retained in `gpu/textures/compressed/pbo.rs`
  with distinct profiles, blocks, decoder oracles, and platform entries.
- **Separate buffer-ticket snapshot helper:** Retained in
  `gpu/execution/buffer_clear_readback.rs`, covering aligned `8..24` and
  indexed unaligned `7..39` snapshots for both buffer usage modes.
- **Unused evidence structs and constructor test:** Removed; they only tested
  a test-only data container and no product test consumed it.
- **Legacy recorder/mock implementation and its 24 self-tests:** Removed; no
  product test used the recorder, which only tested its own simulated trace.
- **Ignored raw GLES indirect driver reproduction:** Removed from crate
  regression targets because it bypassed RHI. Public indirect workloads retain
  the product behavior checks.

Private FFI offset/size checks, real native dispatch counters, state-restoration
checks and pure lowering/admission tests remain independent of GPU readbacks.
An integration case with a similar name is not sufficient grounds to remove them.
