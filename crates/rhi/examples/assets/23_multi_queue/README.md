# Microsoft nBody Gravity assets

The two `official_*.hlsl` files are unchanged from Microsoft
[DirectX-Graphics-Samples](https://github.com/microsoft/DirectX-Graphics-Samples/tree/e5975f9b0744fc5096dd593ed72bf8f5c004164f/Samples/Desktop/D3D12nBodyGravity/src),
commit `e5975f9b0744fc5096dd593ed72bf8f5c004164f`, under the accompanying MIT license.

`compute.hlsl` retains the gravity algorithm and fixes its parameters to the
sample's 10,000 particles, 79 tiles, timestep 0.1 and damping 1; UAV `u0`
is remapped to `u1` to match the public binding number. `draw.hlsl` expands
billboards in an instanced vertex shader because the public raster API has no
geometry shader. It uses the sample's fixed camera, particle radius, color,
radial alpha and additive blend. `initial.bin` is the byte-for-byte initial particle buffer exported from the
original sample's `CreateParticleBuffers` after seeded initialization. Its
SHA-256 is `1ec158493738ab3e3e4c53bf5bc60a2548389f3eaba96e1349f5d12f8f25c6d4`.

Generate the DXIL with `dxc -T cs_6_0 -E CSMain -Fo compute.dxil compute.hlsl`,
`dxc -T vs_6_0 -E VSMain -Fo draw_vs.dxil draw.hlsl`, and
`dxc -T ps_6_0 -E PSMain -Fo draw_ps.dxil draw.hlsl`.
The sample's original asynchronous worker and this example advance simulation
at different rates, so matching wall-clock screenshots are not pixel baselines.
