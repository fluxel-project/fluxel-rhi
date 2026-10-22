# Advanced RHI evidence: 0.17 candidate

The [source-bound verification packet](frozen-cbb7070/README.md) records GPU tests, 16–23 window runs, and readback comparisons on commit `cbb7070c9c69fe90ea5e762cf8d1bb29e8643a8a`.

The later [example 19 fixed-sequence packet](demo19-final-query/README.md) records three DX12 and Vulkan frames after smoke blending, mip sampling, opaque-query flags, and live resource reuse changes. It retains executable and source hashes. The particle states agree with the instrumented NVIDIA reference, while secondary reflected smoke still differs visibly; this is not an exact visual recreation. The [live-window packet](demo19-live-reuse/README.md) separately records changing frames with retained GPU resources.

The [Metal static packet](metal-static-7e98d9e/README.md) binds Core/Metal tests
and both Apple MSRV target checks to commit `7e98d9ebbf03685d76cb412aa625af1ed5a2257b`.
It contains no Apple GPU execution evidence.

The [secondary particle probe](demo19-secondary-probe/README.md) retains a
source-bound NVIDIA shader instrument and screenshot for the original DX12
fixed frame 360. No secondary particle candidate is visible in the unobscured
view on the tested AMD GPU. This does not establish why the RHI adaptation
shows secondary smoke; that visual mismatch remains open.

The [native particle AABB packet](demo19-native-aabb/README.md) records the
later DX12/Vulkan example 19 candidate with particle AABB BLAS and a shared
TLAS queried by both opaque and procedural rays. It passed fixed-frame and
live-window runs, but reflected smoke differs more from the original image.
The packet reports the measured difference rather than claiming pixel parity.

The subsequent [native triangle packet](demo19-native-triangles/README.md)
replaces primary software particle intersections with a dynamic triangle BLAS
and mask-2 RayQuery. DX12 and Vulkan passed fixed-frame and live-window runs;
the reflected-smoke discrepancy and lack of pixel parity remain open.

The [reference stride experiment](demo19-reference-stride/README.md) then
isolated the major discrepancy on the tested AMD GPU. The NVIDIA sample
passes AABB stride 0; changing only that stride to 24 bytes made secondary
particle candidates visible. Comparing the normal shader's corrected
fixed frames to the native triangle/AABB RHI output yields mean RGB errors
of 0.885, 0.949 and 0.855 outside the UI. DXR permits stride 0, so this
observation does not assign responsibility to the driver or sample.

The remaining files in this directory are **pre-freeze working-tree evidence**. The source tree was based on `0b70fa880a44b588efe3758044e604aa3cf7c855` and continued to change during collection. Those screenshots, builds and tests cannot be attributed to a single final commit. `files.json` records SHA-256, byte size and the original local path of each copied artifact; paths inside copied reports refer to the local archive and are retained as historical provenance.

Windows runs used an AMD Radeon 780M, driver 32.0.21028.2002; Vulkan reports driver 25.10.28.02 (LLPC), Vulkan 1.4.315. The Vulkan focused log contains validation-enabled real GPU fixtures; it is not a substitute for the final candidate. Full resolution readback PPMs, unmodified reference captures, executables, build logs and the detailed capture inputs remain in the local verification and evidence directories at the original paths in `files.json` and the copied reports. Contact or window images alone do not establish readback equality.

## Reference and observations

The Microsoft reference sources use [DirectX-Graphics-Samples](https://github.com/microsoft/DirectX-Graphics-Samples/tree/e5975f9b0744fc5096dd593ed72bf8f5c004164f) revision `e5975f9b0744fc5096dd593ed72bf8f5c004164f` and its MIT license. Example 19 uses [NVIDIA Donut Samples](https://github.com/NVIDIAGameWorks/donut-samples/tree/d766a7697b67b787714c3f8837fb271583306ab7) revision `d766a7697b67b787714c3f8837fb271583306ab7` and its included MIT license. Copies of applicable license files are in `licenses/`. Reference readback or fixed-input patches are instrumentation, not claims that the modified executables are unmodified upstream builds.

| Example | Reference and candidate evidence | Scope and limits |
| --- | --- | --- |
| 16 ray pipeline | `visual/16/`; `reports/16-vulkan-dx12-comparison.json` | Vulkan versus DX12 readback, 1280 × 720: 53 pixels differ by one red level. |
| 17 timestamp and statistics | `visual/17/`; `reports/vulkan-final-results.json` | The displayed HelloTriangle readback matches across DX12 and Vulkan; ordered timestamps and statistics are exercised by GPU tests. |
| 18 mesh | `visual/18/`; `reports/vulkan-examples-manifest.json` | ToyRobot readback matches the DX12 reference byte for byte in the candidate run. |
| 19 ray query | `visual/19/`; `reports/19-vulkan-dx12-comparison.json`; `demo19-final-query/`; `demo19-live-reuse/` | Three NVIDIA controlled captures and three fixed-sequence outputs each from DX12 and Vulkan, plus live-window animation. The adapted scene approximates procedural particle intersections; secondary reflected smoke differs visibly from the reference. See the example README and later packet. |
| 20 GPU draw count | `visual/20/`; `reports/vulkan-final-results.json` | Fixed frame 180 from the ExecuteIndirect reference; candidate GPU count is 452 and readbacks match across APIs. |
| 21 sparse texture | `visual/21/`; `reports/21-*`; `logs/21-*-cycle.log` | Nine fixed mip readbacks were compared with the instrumented Microsoft reference on DX12; Vulkan fixed mips match DX12. Screenshots and cycle logs record same-process mip transitions. One contaminated DX12 capture is excluded. |
| 22 texture clear | `visual/22/`; `reports/vulkan-final-results.json` | Uses the reference triangle display; the candidate clears the sampled texture and its readback matches between DX12 and Vulkan. |
| 23 multi queue | `visual/23/`; `reports/23-*` | Three dynamic screenshots per route. Controlled first-step Vulkan versus DX12 differs at 377 pixels by at most one channel level. The DX12 reference was instrumented to freeze a single step. The application waits on the CPU after each presented frame; this demonstrates cross-queue ordering, not overlapping execution. |

The copied Vulkan manifest lists additional per-example source, executable and artifact hashes and explicitly qualifies its moving-tree status. Example 19 runs at 1000 × 709 with controlled states at 8.363, 10.212 and 12.228 seconds; other controlled readbacks use 1280 × 720 where documented by their reports. Screen captures are representative observations and should not be treated as a precise frame number unless their corresponding source manifest states one.

## Reproduction and remaining gate

GPU fixture commands, shader generation checks, original capture commands, environment, diagnostics and expected results are retained in the local candidate logs and manifests. The Microsoft reference is an external checkout; the per-example asset README describes copied or adapted shaders and source. The NVIDIA source, capture script and reference images remain in the local archive at the source paths recorded in `files.json` and in the example README. Preserve those local inputs until a frozen-candidate archive supersedes them.

The source-bound packet adds exact-source rebuilds, affected GPU readbacks, surface runs, image review and diagnostics. The series still lacks Apple GPU acceptance and complete Metal mappings, so this packet is not a full 0.17 release claim.
