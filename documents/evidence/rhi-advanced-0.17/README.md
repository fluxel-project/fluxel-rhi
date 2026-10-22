# Advanced RHI evidence: 0.17 candidate

This is a **pre-freeze working-tree evidence packet**. The source tree was based on `0b70fa880a44b588efe3758044e604aa3cf7c855` and continued to change during collection. The screenshots, builds and tests here cannot be attributed to a single final commit. Before release, rerun the affected checks on the frozen candidate, record its exact commit and refresh this packet. `files.json` records SHA-256, byte size and the original local path of each copied artifact; paths inside copied reports refer to the local archive and are retained as historical provenance.

Windows runs used an AMD Radeon 780M, driver 32.0.21028.2002; Vulkan reports driver 25.10.28.02 (LLPC), Vulkan 1.4.315. The Vulkan focused log contains validation-enabled real GPU fixtures; it is not a substitute for the final candidate. Full resolution readback PPMs, unmodified reference captures, executables, build logs and the detailed capture inputs remain in the local verification and evidence directories at the original paths in `files.json` and the copied reports. Contact or window images alone do not establish readback equality.

## Reference and observations

The Microsoft reference sources use [DirectX-Graphics-Samples](https://github.com/microsoft/DirectX-Graphics-Samples/tree/e5975f9b0744fc5096dd593ed72bf8f5c004164f) revision `e5975f9b0744fc5096dd593ed72bf8f5c004164f` and its MIT license. Example 19 uses [NVIDIA Donut Samples](https://github.com/NVIDIAGameWorks/donut-samples/tree/d766a7697b67b787714c3f8837fb271583306ab7) revision `d766a7697b67b787714c3f8837fb271583306ab7` and its included MIT license. Copies of applicable license files are in `licenses/`. Reference readback or fixed-input patches are instrumentation, not claims that the modified executables are unmodified upstream builds.

| Example | Reference and candidate evidence | Scope and limits |
| --- | --- | --- |
| 16 ray pipeline | `visual/16/`; `reports/16-vulkan-dx12-comparison.json` | Vulkan versus DX12 readback, 1280 × 720: 53 pixels differ by one red level. |
| 17 timestamp and statistics | `visual/17/`; `reports/vulkan-final-results.json` | The displayed HelloTriangle readback matches across DX12 and Vulkan; ordered timestamps and statistics are exercised by GPU tests. |
| 18 mesh | `visual/18/`; `reports/vulkan-examples-manifest.json` | ToyRobot readback matches the DX12 reference byte for byte in the candidate run. |
| 19 ray query | `visual/19/`; `reports/19-vulkan-dx12-comparison.json` | Three NVIDIA original controlled captures and three outputs each from DX12 and Vulkan. Vulkan compares to the adapted DX12 output, not directly byte-for-byte to the NVIDIA executable. The adapted scene approximates the source and presents three precomputed GPU states; it is not continuous simulation. The original and adapted views differ in rendering details; see the example README. |
| 20 GPU draw count | `visual/20/`; `reports/vulkan-final-results.json` | Fixed frame 180 from the ExecuteIndirect reference; candidate GPU count is 452 and readbacks match across APIs. |
| 21 sparse texture | `visual/21/`; `reports/21-*`; `logs/21-*-cycle.log` | Nine fixed mip readbacks were compared with the instrumented Microsoft reference on DX12; Vulkan fixed mips match DX12. Screenshots and cycle logs record same-process mip transitions. One contaminated DX12 capture is excluded. |
| 22 texture clear | `visual/22/`; `reports/vulkan-final-results.json` | Uses the reference triangle display; the candidate clears the sampled texture and its readback matches between DX12 and Vulkan. |
| 23 multi queue | `visual/23/`; `reports/23-*` | Three dynamic screenshots per route. Controlled first-step Vulkan versus DX12 differs at 377 pixels by at most one channel level. The DX12 reference was instrumented to freeze a single step. The application waits on the CPU after each presented frame; this demonstrates cross-queue ordering, not overlapping execution. |

The copied Vulkan manifest lists additional per-example source, executable and artifact hashes and explicitly qualifies its moving-tree status. Example 19 runs at 1000 × 709 with controlled states at 8.363, 10.212 and 12.228 seconds; other controlled readbacks use 1280 × 720 where documented by their reports. Screen captures are representative observations and should not be treated as a precise frame number unless their corresponding source manifest states one.

## Reproduction and remaining gate

GPU fixture commands, shader generation checks, original capture commands, environment, diagnostics and expected results are retained in the local candidate logs and manifests. The Microsoft reference is an external checkout; the per-example asset README describes copied or adapted shaders and source. The NVIDIA source, capture script and reference images remain in the local archive at the source paths recorded in `files.json` and in the example README. Preserve those local inputs until a frozen-candidate archive supersedes them.

The freeze gate still needs exact-source rebuilds and affected real GPU readbacks, surface runs, reviewed screenshots, diagnostic capture, reproducible reference commands, source and executable hashes, stable/MSRV workspace checks, independent review and final results bound to the frozen SHA. Metal has no Apple GPU acceptance in this packet.
