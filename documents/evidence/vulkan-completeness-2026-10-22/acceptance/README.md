# Vulkan acceptance record

The final source candidate is `0e437e0fd95d848e56434b194016bd12aae50f8a`.
Evidence below identifies the actual commit for each run; results from an
earlier commit are not represented as a full rerun on the final candidate.
The production candidate `69d8c2008e643eea5f7afe449a554274f9dc5f68` adds
Unix WSI to `4b87e3f3269d847996609dff8094beb7e2e711bb`. The intermediate
`ffc5487e85b3b8220b977289639d640530d1bc4d` fixes the mesh test fixture; the
final source fixes example shader contracts. The per-run JSON records exact
commands, environment, commit, dirty state, exit status and output hashes.

## Test gates

| Gate | Source and result | Evidence |
| --- | --- | --- |
| Windows stable workspace check | `4b87e3f`: passed. | [metadata](logs/gates/windows-stable-check.json), [stderr](logs/gates/windows-stable-check.stderr) |
| Windows Rust 1.87 workspace library tests | `4b87e3f`: 301 passed, 2 ignored, 0 failed. | [metadata](logs/gates/windows-msrv-workspace.json), [stdout](logs/gates/windows-msrv-workspace.stdout), [stderr](logs/gates/windows-msrv-workspace.stderr) |
| Linux Rust 1.87 workspace library tests | `4b87e3f`: 266 passed, 0 failed on WSL Ubuntu. | [metadata](logs/gates/wsl-msrv-workspace.json), [stdout](logs/gates/wsl-msrv-workspace.stdout), [stderr](logs/gates/wsl-msrv-workspace.stderr) |
| Linux stable Vulkan all-target check | `4b87e3f`: passed; compiles the Unix opaque-FD route. | [metadata](logs/gates/wsl-stable-vulkan.json), [stderr](logs/gates/wsl-stable-vulkan.stderr) |
| Linux Rust 1.87 affected Vulkan check | `69d8c20`: all targets with hardware-tests passed. | [metadata](logs/gates/wsl-final-msrv-vulkan.json), [stderr](logs/gates/wsl-final-msrv-vulkan.stderr) |
| Unix surface fixture | `69d8c20`: Xlib and XCB presentation passed on llvmpipe (LLVM 20.1.2), one test. This is software execution, not GPU or Wayland presentation evidence. | [metadata](logs/gates/wsl-final-unix-surface.json), [stdout](logs/gates/wsl-final-unix-surface.stdout), [stderr](logs/gates/wsl-final-unix-surface.stderr) |
| Vulkan hardware suite | `4b87e3f`: 78 tests passed including 10 unit tests. Three mesh depth tests emitted validation VUID 09588 in the original run; their passing status alone does not close that diagnostic. | [metadata](logs/gates/vulkan-gpu.json), [stdout](logs/gates/vulkan-gpu.stdout), [stderr](logs/gates/vulkan-gpu.stderr) |
| Mesh fixture correction | `ffc5487`: all three affected mesh tests passed with validation enabled and no VUID 09588 after the fixture's depth transition was corrected. | [metadata](logs/gates/mesh-final.json), [stdout](logs/gates/mesh-final.stdout), [stderr](logs/gates/mesh-final.stderr) |
| Core interop lifetime doctest | `ffc5487`: compile-fail borrow test passed. | [metadata](logs/gates/core-docs.json), [stdout](logs/gates/core-docs.stdout) |
| DX12 sparse contract regression | `0e437e0`: four affected hardware tests passed with debug diagnostics enabled. | [metadata](logs/gates/dx12-sparse-regression.json), [stdout](logs/gates/dx12-sparse-regression.stdout), [stderr](logs/gates/dx12-sparse-regression.stderr) |

The Windows device was AMD Radeon 780M Graphics, AMD proprietary driver
25.10.28.02, Vulkan driver version 2.0.353; the full device information is in
[device output](logs/gates/device.stdout). The runs have compiler warnings,
including Vulkan binding retention fields, and do not imply warning-free builds.
Independent reviews reported no remaining confirmed P0/P1 in their reviewed
scope; this does not turn unavailable hardware paths into tested paths.

## Examples and regression

All 23 Vulkan examples have a successful process run in [vulkan logs](logs/vulkan/).
Examples 05 and 13 initially failed because their shader declarations and
pipeline layouts disagreed. Their [initial 05](logs/diagnostics/initial-vulkan/05_push_constants/run.stdout)
and [initial 13](logs/diagnostics/initial-vulkan/13_multithreaded_recording/run.stdout)
diagnostics are retained. After the example contract correction, 05, 13 and
14 were rebuilt and rerun at `0e437e0`; the [build metadata](logs/gates/examples-final-rebuild.json)
records that scope. Examples 03, 04, 07–12, 15 and 21 use reruns at `69d8c20`;
the remaining successful example runs use `ffc5487`. The corresponding 13
changed examples also have successful [DX12 regression logs](logs/dx12/), with
05, 13 and 14 at `0e437e0` and the other ten at `69d8c20`.

Example 19 records three in-app readbacks rather than a WSI screenshot. Process
success or presentation alone is not a visual correctness assertion. The
separately maintained visual evidence and image manifest describe captured
frames, inspection and any limitations.

## Visual inspection

[The contact sheet](visual/contact.png) contains all 23 Vulkan scenes.
[The image manifest](visual/manifest.json) records 33 Vulkan images and two
DX12 client-window captures, source commits, capture method, dimensions and
hashes. PPM captures were converted losslessly to PNG; decoded RGB bytes were
compared before archiving. Each Vulkan image was checked for constant output.
Example 19 uses its own three GPU readbacks at the times printed in its log.
Most other images use the screenshot layer's frame 2; examples 11, 17, 22 and
23 also retain frames 30 and 60. The static 11, 17 and 22 images are not motion
evidence.

The primary reviewer opened the contact sheet and full-size selected captures.
An independent reviewer opened the contact sheet, full-size Vulkan and DX12
05/13, all three 19 readbacks, 11 frames 2/60, 23 frames 2/60 and 18. The
inspected views contain recognizable scene geometry, including the colored
ring, ship fleet, reflective objects and smoke, robot, and changing particles.
No blank scene or obvious missing geometry was found. An initial claim that
the DX12 ring capture was black was retracted after reopening the exact file
and verifying its SHA-256. The remaining scenes were reviewed as thumbnails;
this is not a C++ reference pixel comparison or exhaustive interactive test.

The DX12 captures were taken one second after launch, with unknown frame
numbers; their [capture metadata](logs/gates/dx12-push-visual.json) and
[secondary capture metadata](logs/gates/dx12-secondary-visual.json) describe
the commands. A snapshot of the client-window capture utility is retained in
[tools](tools/capture_demo.py). Animation timing and inputs were not synchronized
between backends, so these images do not establish frame-identical rendering.

## Limits of this evidence

- Device feature, format, usage, queue and extension checks still govern each
  descriptor. ETC2/EAC and ASTC were unavailable on the tested GPU. Sparse
  depth formats and shared-tail/multi-aspect paths require supporting-device
  execution; the recorded array tail was not shared.
- OPAQUE_WIN32 external-memory and binary-semaphore fixtures run on Windows;
  the OPAQUE_FD path was compiled on Linux, without an external FD GPU round
  trip. The producer must supply compatible allocation and synchronization.
- The separate pass-boundary statistics domain is not implemented by Vulkan;
  its feature-gated eleven-field pipeline statistics query is a different API.
  Native interop requires the caller to restore state and retain resources.
- This record does not claim Apple GPU execution. Mock tests, compilation,
  process exit, and a software presentation smoke each establish only their
  stated scope. Earlier full workspace gates were not repeated on the final
  example-only correction.

Raw logs are byte-preserved and indexed by [SHA256SUMS](SHA256SUMS). A final
artifact manifest including visual files is maintained separately.
