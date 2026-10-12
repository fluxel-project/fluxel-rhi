# RHI conformance matrix

**Scope:** current checkout. The only facade-exposed native provider is DX12 on Windows.
GL, Vulkan, Metal and WebGPU implementation/test paths are not present in this checkout
and therefore have no matrix row.

`Pass` means a recorded native run, not successful compilation. `Unsupported` requires a
published capability or operation result. `Not yet evidenced` makes no claim either way.

| Route | DX12 status | Evidence / command |
| --- | --- | --- |
| Typed Core/facade integration | Checked | `cargo test -p fluxel-rhi-core`; facade examples compile through `fluxel_rhi::api` |
| Resource, DXIL shader and raster pipeline creation | Checked | numbered DX12 examples and `cargo check -p fluxel-rhi --features examples` |
| Surface acquire, graphics submit and present | Checked on Windows | numbered window examples |
| RGBA8 textures, views, sampler bindings and uploads | Checked | texture examples; descriptor-specific support remains authoritative |
| Depth attachments | Checked | depth/offscreen/MSAA example routes |
| Texture-array SRV and KTX mip/layer upload | Hardware pass | `10_indirect_draw -- --frames 2` |
| Indexed indirect draw (`ExecuteIndirect`) | Hardware pass | `10_indirect_draw -- --frames 2` |
| Multi-draw | Capability-gated | depends on `Device::capabilities()` |
| Push constants | Capability-gated | no general hardware-pass claim |
| Occlusion query and resolve to readback | Hardware pass | `11_occlusion_query -- --frames 2`; observed 50112 and 0 passing samples |
| Buffer/texture readback | Hardware pass | `12_screenshot -- --frames 2`; generated and inspected 1280×720 PPM |
| Secondary raster work | Hardware pass | `13_multithreaded_recording -- --frames 2` |
| Descriptor indexing | Hardware pass | `14_descriptor_indexing -- --frames 2` |
| Two-view multiview | Hardware pass | `15_multiview -- --frames 2` with DX12 debug layer |
| Compute and mip generation | Descriptor/capability-gated | test each selected route before recording a pass claim |

The two hardware-pass cells above describe the local finite-frame run performed for the
current DX12 route. Re-run it after changes to DX12 command lowering, descriptors,
pipelines, texture arrays, indirect recording, or the example assets.

## Updating this matrix

Add a row only after recording a reproducible command and result. A new backend first
needs a checked-in implementation, facade exposure, accurate capability reporting and a
native execution test; a planned provider belongs in design notes, not this matrix.
