# Secondary particle candidate probe

The source patch instruments NVIDIA Donut Samples revision
`d766a7697b67b787714c3f8837fb271583306ab7`, using the fixed frame
C++ patch in `../demo19-fixed-sequence/`. It returns opaque red immediately
when a secondary particle ray query yields any candidate. Build the
`rt_particles_shaders` Release target, set `FLUXEL_REFERENCE_FRAME=360` and
`FLUXEL_REFERENCE_STATE` to an output file, launch the DX12 executable,
resize its client area to 1000 x 709, and capture after it pauses. The patched
shader, shader package, screenshot and executable hash are retained.
The copied upstream shader and patch retain their original whitespace so their
recorded hashes remain tied to the compiled package.

On AMD Radeon 780M (driver 32.0.21028.2002), no red pixels appeared in
the visible client area. Compared to `../demo19-fixed-sequence/` frame 360,
the captured RGB mean absolute difference is below 0.00025 on each channel;
the original UI occludes part of the viewport. This establishes only that no
secondary candidate became visible in the unobscured screenshot for this run;
it does not prove why. Different captures can contain tiny screenshot noise.
The original trace uses mask 4, no ray flags, and limits TMax to the opaque
intersection. The RHI example does not build the particle AABB TLAS; it uses
software billboard intersections. This semantic difference remains unresolved.
