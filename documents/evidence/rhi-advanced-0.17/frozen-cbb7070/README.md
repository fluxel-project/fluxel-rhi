# Source-bound advanced RHI verification

The exact implementation source was commit `cbb7070c9c69fe90ea5e762cf8d1bb29e8643a8a` on Windows MSVC, AMD Radeon 780M, driver 32.0.21028.2002 (Vulkan driver 25.10.28.02). The evidence documentation in this directory was added after that source commit without changing the implementation. `manifest.json` gives SHA-256 digests for each retained log and PNG, plus the original full-resolution PPM digests.

## Commands and results

- `cargo +stable check --workspace --all-targets` and `cargo +1.87 check --workspace --all-targets --quiet`: both passed.
- `cargo check -p fluxel-rhi-metal --target aarch64-apple-darwin --quiet` and the `x86_64-apple-darwin` equivalent: both passed. These are compile checks, not Apple GPU execution.
- `cargo test -p fluxel-rhi-core -p fluxel-rhi-metal --lib --quiet`: 51 Core and 49 Metal tests passed.
- `cargo test -p fluxel-rhi-dx12 --features hardware-tests --tests --quiet`: all test binaries passed on the local GPU; retained in `logs/dx12-final-gpu.log`.
- `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation cargo test -p fluxel-rhi-vulkan --features hardware-tests --tests --quiet`: all test binaries passed on the local GPU; retained in `logs/vulkan-final-gpu.log`. The captured logs contained no validation errors.
- For each example 16–23 and each `dx12`/`vulkan` route: `cargo run -p fluxel-rhi --features examples,dx12,vulkan --example NAME -- --backend ROUTE --frames 2`. All 16 runs exited successfully. The per-route logs are retained in `logs/`; the Vulkan environment enabled `VK_LAYER_KHRONOS_validation`.

Examples 16, 17, 18, 20, 21, 22, and 23 were run again with their documented `FLUXEL_*_READBACK` path set. Example 21 fixed `FLUXEL_SPARSE_MIP=0`; the separate GPU tests exercise sparse switching. Each GPU readback was 1280 × 720 RGB and converted without pixel modification to `readback/*.png`. A visual contact sheet of all 14 images was opened and reviewed: the expected triangle, robot, indirect field, checkerboard, cleared triangle, and particle cloud were visible with matching DX12/Vulkan composition. Example 19 emitted three controlled 1000 × 709 PNG states on each route during the source-bound runs. A separate contact sheet was opened and reviewed: smoke position and color change across states, and each DX12/Vulkan state has matching composition. The NVIDIA original comparison remains in the pre-freeze packet; the candidate scene is an approximation, not pixel-identical to that original.

| Example | Different RGB channels, DX12 vs Vulkan | Maximum channel delta |
| --- | ---: | ---: |
| 16 ray pipeline | 53 | 1 |
| 17 GPU queries | 0 | 0 |
| 18 mesh | 0 | 0 |
| 19 ray query states 1/2/3 | 1388 / 1464 / 1447 | 25 / 25 / 25 |
| 20 GPU draw count | 0 | 0 |
| 21 sparse mip 0 | 0 | 0 |
| 22 clear texture | 0 | 0 |
| 23 multi queue first frame | 390 | 1 |

These comparisons establish the exercised device and inputs, not universal feature support. Windows GPU execution does not establish Metal support. Metal sparse texture binding, exact pipeline statistics, GPU-count multi-draw, and a full ray pipeline remain unadvertised; see the root README and design document for route limits.
