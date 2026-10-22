# HelloTriangle shader source

`shaders.hlsl` is copied from Microsoft's [D3D12HelloTriangle](https://github.com/microsoft/DirectX-Graphics-Samples/tree/e5975f9b0744fc5096dd593ed72bf8f5c004164f/Samples/Desktop/D3D12HelloWorld/src/HelloTriangle), distributed under the included MIT license. Its SHA-256 is `c300fc6586b8e0410ea3df4c00d080efeeeeac7e0e2485a84864fae118f6597e`.

`shaders_rhi.hlsl` changes only the vertex input semantics from `POSITION` and
`COLOR` to `LOCATION0` and `LOCATION1`, matching the DX12 RHI input layout.
Its SHA-256 is `4a7d420a0dda29ae9bc8ba446528fadd480895c0ad657d588a1cd971e33d913d`.
The shader calculations and pixel output remain unchanged. Generate the DXIL:

```powershell
dxc -T vs_6_0 -E VSMain -Zi -Qembed_debug -Fo triangle_vs.dxil shaders_rhi.hlsl
dxc -T ps_6_0 -E PSMain -Zi -Qembed_debug -Fo triangle_ps.dxil shaders_rhi.hlsl
```

The example reads two timestamps and one pipeline statistics record after
completion. It checks the timestamps, queue frequency, three input vertices
and one input primitive. Set `FLUXEL_QUERY_READBACK` to save the rendered RGB
PPM and compare it with the official GPU output using
`tools/examples/compare-ray-hello.py`.

Timestamp placement and frequency follow the use of native timestamps in
Microsoft's [GpuTimeManager](https://github.com/microsoft/DirectX-Graphics-Samples/blob/e5975f9b0744fc5096dd593ed72bf8f5c004164f/MiniEngine/Core/GpuTimeManager.cpp).
