# Example 19: fixed sequence after opaque-query correction

The candidate source and executable hashes are in `sha256.json`. The example was
built on Windows MSVC with an AMD Radeon 780M, driver 32.0.21028.2002. Each
backend was run with `FLUXEL_RAY_QUERY_FIXED_REFERENCE=1` and
`FLUXEL_RAY_QUERY_STATE_DIR` pointing at its output directory. DX12 used
`FLUXEL_DX12_DEBUG=1`; Vulkan used `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`.
Both returned zero with empty stderr. Images are 1000 x 709 at frames 120,
240 and 360, with deterministic 1/60 second simulation steps. The frame 360
DX12 capture was opened and checked: scene, smoke and reflections are visible.

The retained NVIDIA fixed-frame reference and the source patch used for its
simulation are in `../demo19-fixed-sequence/`. Comparing the region
(400, 0, 1000, 709) to NVIDIA, mean RGB absolute errors of the DX12 candidate
are 4.220, 4.029 and 4.845, compared with 4.250, 4.055 and 4.868 in the
previous candidate. Vulkan has the same rounded errors. These metrics do not
establish visual equivalence: reflected smoke remains conspicuously different.
The original uses a mask-4 procedural particle ray query for secondary rays;
the example still uses software plane intersections. The opaque query now
uses back-face culling and TMin 0, matching the source's corresponding flags.

The `nvidia*.png` and `diagnostic-primary-*.png` files record exploratory
instrumentation, not the final build. In the fixed frame 360 NVIDIA run,
disabling secondary particle blending and marking secondary candidates red
both produced an image equal to the original capture; disabling all particle
blending removed the smoke. The RHI diagnostic images disabled secondary
particle blending and made the primary smoke location much closer. Those
instrumented binaries were not retained, so the diagnostic images do not
constitute a reproducible source-bound proof of the cause. The final example
keeps secondary smoke enabled; its unresolved difference is not hidden.
