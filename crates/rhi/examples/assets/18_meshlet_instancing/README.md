# Microsoft MeshletInstancing assets

These files originate from Microsoft DirectX-Graphics-Samples, revision `e5975f9b0744fc5096dd593ed72bf8f5c004164f`, under the included MIT license. `ToyRobot.bin` is from `Samples/Desktop/D3D12MeshShaders/src/Assets`; the shader sources and `Shared.h` are from the adjacent `MeshletInstancing` directory.

`MeshletMS.rhi.hlsl` changes only shader resource registers t0–t4 to t3–t7 and removes the embedded root signature attribute to use the RHI pipeline layout. The computation is unchanged. The original source is retained. The DXIL files were produced with DXC 1.9.0.5399:

```text
dxc -T ms_6_5 -E main -Fo MeshletMS.rhi.dxil MeshletMS.rhi.hlsl
dxc -T ps_6_0 -E main -Fo MeshletPS.dxil MeshletPS.hlsl
```

SHA-256:

- `ToyRobot.bin`: `baaee3fdaa7ad17f78416b4f9a4a4c7812cca1323ba37fe8e34a68d29fc2dd7e`
- `MeshletMS.hlsl`: `a1ab0e20d2beff5ab410ec694dd4b9cbbad120ff803c48365818ae3eceea0126`
- `MeshletPS.hlsl`: `605c99319b299c286e835418600ff6c6a19a589d9b78047ebf2b70ad58d8d49e`
- `Shared.h`: `f7865fe8824233ce6e36ff41ce4ac361409bd5d154220e03ec310a465de7ddf8`
- `MeshletMS.rhi.hlsl`: `7cb26df6a8a64eee4da6169bde4e1973f8172be27b257082982f28eb7f9f9c20`
- `MeshletMS.rhi.dxil`: `1c5e8319ddcd27a17f45b2fa4f2325e54d4c875b591dd6d9e536408cf1d891ee`
- `MeshletPS.dxil`: `292ce1ae6d1f1dde7c9fa5c657aefc766d25dd32278ee4dd0a5c7fdf83bd686f`

`MeshletMS.vulkan.hlsl` adapts the RHI mesh shader for Vulkan: draw parameters
use a push constant and packed vertex data is loaded from a byte address buffer
with the same 24-byte stride. This avoids requiring scalar block layout. The
scene calculations are unchanged. The repository script
`tools/examples/build-native-spirv.py --verify` rebuilds and validates SPIR-V.
