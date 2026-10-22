# Rebuilt NVIDIA reference and trajectory attempt

Portability note: the retained helper scripts now take their external reference
checkout from `FLUXEL_DONUT_REFERENCE_ROOT`. Manifest entries with that prefix
refer to the local reference checkout. This changes no retained image or native
artifact; the original capture script hash was
`e4f08dbb6730b251fe351c26df3eb9c09749c9d3d01674762e8e3b2a3cd9d195`.

The stride-24 reference C++ was restored from `../demo19-reference-stride/instrumented-stride24.cpp` and rebuilt in Release using the existing Visual Studio 2022 CMake tree, target `rt_particles`. `sha256.json` records source, normal shader, executable and capture hashes. The C++ and shader hashes match the retained reference artifacts; the rebuilt executable hash differs. On AMD Radeon 780M with driver 32.0.21028.2002, `../demo19-reference-stride/capture.py 360` produced `rebuilt-360.png` at 1000 x 709. The 360-frame state dump has SHA256 `11ba9bfe39e0d126f9106cd83c7888fae64d0af15a9537c00a27c9c09d979084`, identical to the retained state. RGB mean absolute error against `../demo19-reference-stride/corrected-360.png`, in (400, 0, 1000, 709), was 0.00382.

`capture.py` attempts the same posted Win32 drag and wheel events as `../demo19-camera-controls/input-probe.py` with fixed-frame simulation. Its `initial.png` was visually obstructed by another window, and pan, Alt-pan and zoom produced no visible changes after orbit. These images do not establish equivalent input delivery or camera trajectory and must not be used as comparison evidence. Capture was terminated after these failures; no conclusion about trajectory fidelity is drawn.
