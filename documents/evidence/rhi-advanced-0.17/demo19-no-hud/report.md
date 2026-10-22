# Example 19 without HUD

Windows MSVC, AMD Radeon 780M, driver 32.0.21028.2002, 1000 x 709. Source and executable SHA-256 values are in sha256.json. DX12 debug and Vulkan validation were enabled. Both live captures show the default HUD-free scene; the DX12 capture was opened for inspection. Both processes remained alive until stopped by the probe and the retained logs contain no diagnostics.

The optional panel is toggled with H. The automated posted keyboard messages did not open it, so the shortcut is covered only by the unit test and requires real keyboard validation. These live captures use real-time simulation and are not time-matched to NVIDIA. The fixed-frame comparison and reference rebuild evidence are recorded separately.
