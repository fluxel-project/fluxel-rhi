# DX12 candidate acceptance

Date: 2026-10-22. Production candidate: `5b9c4d1c18f496ed5d3ed57494beb37537ca29cf`.
The following evidence is restricted to the tested device, descriptors and inputs.

## Environment

- Windows 11, AMD Radeon 780M Graphics, driver `32.0.21028.2002` (2025-09-04).
- Windows target `x86_64-pc-windows-msvc`; MSRV Cargo/Rust 1.87 and stable Cargo 1.98.
- WSL2 Ubuntu, Rust 1.87.0, `x86_64-unknown-linux-gnu`, for portable library tests.
- DX12 debug layer enabled with `FLUXEL_DX12_DEBUG=1`.
- `FLUXEL_NATIVE_COSTS_SMOKE=1` makes the diagnostic workload a correctness check;
  these debug runs are not performance measurements. See the separate [A/B report](../performance.md).

## Gates and revisions

| Gate | Result and exact scope |
| --- | --- |
| Stable workspace check | `cargo +stable check --workspace --all-targets --all-features` passed at `a13899315963ad9b25c42ff3e8f28ea163492bbd`. |
| Windows MSRV library tests | `cargo +1.87 test --workspace --lib`: 274 passed at `5033fe9`. The earlier shader capability witness failed because it advertised ray query with an unavailable Ray implementation; the corrected witness implements that interface. Production Core behavior was retained. |
| Linux MSRV library tests | WSL2 `cargo +1.87.0 test --workspace --lib`: 244 passed at `5033fe9`. |
| DX12 complete suite | `cargo +1.87 test -p fluxel-rhi-dx12 --features hardware-tests -- --test-threads=1`: 228 passed, including 34 backend unit tests, 13 shader compiler tests and 181 hardware fixture tests. Initial run at `a138993`; diagnostic rerun with `--show-output` at `5033fe9`. |
| Final affected tests | On `5b9c4d1`, nine sparse binaries passed 11 tests and the two depth fixtures passed two tests, with no D3D12 severity 0/1 messages. The final stable DX12 all-target check passed. See `logs/sparse-final.log`, `logs/affected-final.log`, and `logs/stable-affected-final.log`. |
| DX12 demos | All 23 numbered examples returned success. The requested frame limit was three; some advanced examples run their own fixed capture sequence. This is execution/presentation smoke, not visual proof for every demo. |
| Changed examples on Vulkan | 10, 11, 17 and 19 returned success. Example 17 also reported valid timestamps and IA counts (3 vertices, 1 primitive). |

Between the complete suite and the final candidate, production changes are
limited to sparse array preflight and its private helper. Two depth fixtures
were corrected to use the actual initial `DEPTH_WRITE` state. The final affected
set covers those changes; the full suite was not repeated after these fixes.
The complete diagnostic log intentionally retains the original seven validation
errors: six incorrect fixture barriers and one sparse array descriptor rejected
by the driver. Their corrected logs supersede those results. They must not be
described as an initially validation-clean run. Clear-value performance warnings
820/821 are retained in the logs.

## Visual inspection

Both the integrator and a separate reviewer opened all eight retained images:

| Images | Observed result |
| --- | --- |
| `01_triangle-01.png`, `17_gpu_queries-01.png` | Complete colored triangles on their expected backgrounds. |
| `04_dynamic_uniform_buffer-01.png` | Multiple colored cubes; no blank frame or obvious missing geometry. |
| `10_indirect_draw-01.png` | Textured forest with palms, shrubs and flowers. |
| `11_occlusion_query-01.png` | Blue occluder, colored sphere and red teapot. |
| `ray-query-particles-01.png` through `-03.png` | Reflective columns, spheres and ground with changing translucent colored smoke. |

The first five window captures were requested two seconds after launch, at
1280x720, with no user input; their exact simulation frame was not recorded.
Example 19 saved 1000x709 scene readbacks at its own fixed capture points, logged
as 8.363, 10.212 and 12.228 seconds with 38, 30 and 30 particles. HUD appearance
was excluded. No new same-frame C++ pixel comparison was performed, so these
images do not establish exact equivalence or all-animation correctness. Query
numbers are checked separately by GPU fixtures and demo diagnostics.

## Review and remaining boundaries

The [original review mapping](../review.md) covers all five numbered findings
and the two additional performance topics. Independent static review checked
submission/failure retention, upload retirement, descriptor heap installation,
query dependency coverage, and compute/ray state restoration. DX12-LIFE-01 was
closed after checking Core's poisoned-encoder rule and the additional native
ownership hardening.

Remaining rejection paths include absent hardware features, invalid usages,
native queue/state/format restrictions and combinations whose portable contract
does not define a result. Examples include partial buffer state transitions,
ETC/ASTC without DXGI formats, writes to compressed textures through blit, and
same-subresource copies requiring incompatible source/destination states.
Cross-numeric and mixed-dimension blit conversions are not claimed. MSAA 16/32
enters the device-query path but this GPU reported only 1/2/4/8 samples. Sparse
packed array tails remain outside the current single-layer tail contract.

This work does not establish RenderGraph integration, Apple GPU behavior,
HDR monitor output, every shader feature, or every device-specific combination.
The final file hash manifest binds retained logs and images; evidence-only
commits after the production candidate do not change the tested implementation.
