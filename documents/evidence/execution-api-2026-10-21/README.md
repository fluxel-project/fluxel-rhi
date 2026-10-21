# Common execution API and example migration evidence

Candidate base: `efc4dbba5812861778f76f257de2d29035eb209e` plus the source
fingerprints in [final-rust.json](sources/final-rust.json). This packet is a
change-specific regression record, not a release or universal capability claim.

## Contract and scope

The optional Execution interface statically selects deferred native recording
or immediate issue. GL calls happen in the encoding closure, before finish.
No CPU instruction stream, replay, scheduler, normal-path idle wait, or per-step
fence was introduced. Mock traces verify native SPI calls before finish and no
checkpoint between command steps; source review checks the native GL lowering.

All 15 examples use the common entry. Deferred helpers bind existing native
recordings without converting commands, preserving example 13 parallel secondary
recording. GL uses the same scene encoding immediately. Sparse and native
multi-queue execution retain their explicit APIs. This work implements no
RenderGraph compilation, hazard analysis, or resource-retirement policy.

## Results

| Verification | Result | Limit |
| --- | --- | --- |
| Core unit tests | 35 passed | Includes immediate timing, partial/uncertain effects, borrowed facts, plan preflight, and recorded-work ownership |
| Example host and external SPI tests | 7 passed | Includes uncertainty retained through the failure-only drain |
| GPU execution fixtures | 5 passed | Legacy DX12/Vulkan plan fixtures plus common DX12/Vulkan/WGL four-step exact readbacks; WGL also clear/present smoke |
| Windows native examples | 45 three-frame runs passed | Fifteen per DX12/Vulkan/WGL; migration-candidate visual capture remains unavailable while the desktop is locked |
| Desktop browser examples | 30 passed, screenshots inspected | Fifteen per WebGPU/WebGL2, three frames at 1280x720, AMD hardware adapter |
| WSL GLES examples | 15 passed, framebuffer captures inspected | Three frames each, Mesa 25.0.7 D3D12 / AMD Radeon 780M, not llvmpipe |
| Android native examples | 30 passed, screenshots inspected | Fifteen per Vulkan/GLES, two frames each; both example 12 routes produced GPU readback |
| Android browser examples | 15 passed, screenshots/readback inspected | Three frames each, WebGL2 in WebView 146; WebGPU adapter unavailable |
| Builds | Workspace all-targets, native examples, browser all-targets, Android 30 target combinations passed | Apple GPU validation and release/MSRV gates were not performed |
| Formatting and review | Formatting/diff checks and independent read-only contract review passed | Standard Core Clippy completed with warnings; no warning-free claim |

The native Windows captures attempted after migration showed the Windows lock
screen and were rejected. They are not copied into this packet. Passing process
exit or headless readback is not substituted for the missing per-example native
visual acceptance. Earlier baseline images remain historical evidence only.
The random initial selection was examples 05 and 13; after the request expanded
to all examples, the execution regression covered every example on each tested
route. Metal remains outside hardware acceptance.

## Candidate provenance

The source manifests identify build candidates and the executable/WASM hashes
identify actual artifacts. The Android report distinguishes its earlier GLES
13 capture from the final source snapshot after comment-only changes. Desktop
Windows smoke ran before three documentation-only edits; a final native build
manifest is retained separately. Browser final build/run fingerprints match;
Android browser used those exact WASM hashes. The later formatting-only change
to the independent host test file does not enter example WASM. Full original
logs, per-run candidates, frame images, APKs and readbacks remain in local
archives; this packet retains representative contact sheets and compact records.

Mock failures establish Core/host behavior, not native driver failure recovery.
The emulator reports NX809J, Android 14 and Adreno 750; that identity is not a
physical-device certification. Vulkan validation cleanliness is not established
because these runs do not install a debug messenger. No performance percentage
or allocation-free deferred submission claim is made.

## Reproduction

```powershell
cargo test -p fluxel-rhi-core --lib
cargo test -p fluxel-rhi --no-default-features --test frame_submission
cargo test -p fluxel-rhi --features vulkan,native-gl-wgl --test unified_execution_gpu --test execution_plans_gpu -- --ignored --nocapture
cargo check --workspace --all-targets
cargo check -p fluxel-rhi --target wasm32-unknown-unknown --no-default-features --features examples,webgpu,webgl2 --all-targets
```

Platform setup and runtime commands are in the [testing guide](../../testing.md),
[example guide](../../../crates/rhi/examples/README.md),
[Android runner guide](../../../tools/examples/android-runner/README.md), and
[Android browser guide](../../../tools/examples/android/browser/README.md).
`files.json` indexes the preserved bytes; `.gitattributes` prevents newline
conversion from invalidating those evidence hashes.
