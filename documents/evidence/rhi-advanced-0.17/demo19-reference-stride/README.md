# NVIDIA reference: particle AABB stride experiment

On Windows MSVC, AMD Radeon 780M, driver 32.0.21028.2002, NVIDIA Donut
Samples revision `d766a7697b67b787714c3f8837fb271583306ab7` was built
in Release mode. The compared RHI source is commit
`f29fb5efe98fd48d813f5bae7c7605f8c4a87ff2`; its per-file source and
executable hashes are in `../demo19-native-triangles/sha256.json`. The patch
`instrumented-stride0.patch` contains the fixed 1/60 step and frame pause,
a TLAS instance dump, and the candidate count shader. The otherwise identical
`instrumented-stride24.patch` adds
`.setStride(sizeof(nvrhi::rt::GeometryAABB))` to the particle AABB geometry.
The complete modified C++ and HLSL, shader packages and executable hashes
are also retained. `capture.py` takes a frame number and captures a
1000 x 709 client area after the fixed-step animation pauses.
The copied upstream source and generated patches retain their original
whitespace so their hashes match the compiled shader packages and captures.

The instrumentation counts procedural candidates on each secondary ray,
then writes red for any count and black for zero directly to the output,
independently of particle blending, path attenuation and the environment.
In the UI-free rectangle (400, 0, 1000, 709), the stride-0 capture and a
repeat have zero red pixels out of 425400; stride 24 has 251670. The TLAS
dumps show 46 instances, including 40 non-null particle BLAS references with
mask 4 and compact IDs 0 through 39. The source binds that TLAS at t0; NVRHI
copies the instance descriptions to an upload allocation and builds the
TLAS from it. The red probe directly observes successful GPU traversal in
the stride-24 case. UI obscures the remaining portion of the screen.

NVRHI defaults `GeometryAABBs::stride` to zero and passes it through to
DX12; the original sample does not set it. DXR permits stride zero, so this
observation does not establish a specification violation or isolate whether
the difference is a driver or sample interaction. It establishes that on
this device and run, changing the stride to 24 changes traversal outcomes.
See the [DXR functional specification](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html).

`corrected-120.png`, `corrected-240.png` and `corrected-360.png` use the
unmodified NVIDIA shader and the stride-24 executable, with fixed-step and
TLAS-dump instrumentation. Compared to the RHI captures in
`../demo19-native-triangles/`, the mean absolute RGB errors in the UI-free
rectangle are 0.885, 0.949 and 0.855 for DX12; Vulkan differs from those
metrics by under 0.00003. The earlier comparison against unmodified stride-0
reference was 5.302, 5.269 and 5.832. The corrected frame 360 was opened:
reflected smoke is visible in both corrected NVIDIA and RHI images. These
are image comparisons on one GPU, not proof of pixel-identical rendering.
