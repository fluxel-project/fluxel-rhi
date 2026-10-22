# DX12 array copy fixture observations

These are development observations, not frozen-candidate acceptance evidence.
The workspace contained uncommitted changes based on commit
`66e77710034dbcb1c8afd458106715f6c67f57e8`; other agents also edited the
workspace during these runs. The exact complete source tree was not frozen.

Environment: Windows, AMD Radeon 780M Graphics, driver 32.0.21028.2002,
`rustc 1.97.0-nightly (507271bc1 2026-05-17)`, DX12 debug layer enabled by
`FLUXEL_DX12_DEBUG=1`. The tests used headless buffer readback.

The following command completed with one passing test per named fixture:

```powershell
$env:FLUXEL_DX12_DEBUG='1'
cargo test -p fluxel-rhi-dx12 --features hardware-tests --test d1_array_copy_gpu --test d1_array_cpu_gpu --test color_array_copy_gpu --test color_array_cpu_gpu --test bc_array_copy_gpu --test bc_array_cpu_gpu --test depth_array_copy_gpu --test depth_array_stride_gpu -- --nocapture
cargo test -p fluxel-rhi-dx12 --features hardware-tests --test depth_array_cpu_gpu -- --nocapture
```

The first command produced eight passing fixtures and the second produced one.
The successful runs reported no D3D12 debug messages. Their console output was
observed in the agent tool transcript; full console logs were not separately
saved. An earlier depth CPU fixture using subnormal floats failed a bitwise
comparison, and was changed to ordinary finite depth values before the passing
run. These results do not establish preservation of subnormal depth bit patterns.

| Fixture | Observed coverage | SHA-256 of fixture at report time |
| --- | --- | --- |
| `d1_array_copy_gpu` | Three-layer D1 buffer/texture round trip | `d75ff0b2d4d45ba656cf33f22204c4ced45aa730f2a574215f6fb62f7e902e60` |
| `d1_array_cpu_gpu` | D1 CPU upload and texture copy/readback | `836fefaa4fccc5ab338b7fef59cd6a9011386bd1b9988cc53e2a5b51172a2692` |
| `color_array_copy_gpu` | RGBA8 three-layer buffer round trip, 768-byte image stride | `2fc6f9ad6847296a33ff82f35339237877d072d717c549298a5dcb720c80ba21` |
| `color_array_cpu_gpu` | RGBA8 CPU upload and texture copy/readback | `e2d7f1cd8c18aee53a61ae5009dd947d0ec9311c70f96f2ecce9961ebcd24e13` |
| `bc_array_copy_gpu` | BC1 three-layer buffer round trip, 768-byte image stride | `befdccd07f9ac4e36477ef7a7de9e45e4f7105f7804cd07e2d4bee16f98c65a2` |
| `bc_array_cpu_gpu` | BC1 CPU upload and texture copy/readback | `fe3403e5fb95737c7c88da701fa83d02de9fdc08e2e981023007511edacdaa3c` |
| `depth_array_copy_gpu` | D24S8 and D32S8 stencil and D32S8 depth plane round trips | `9e2f027d4a6ad9800d224720203783b03f6af0e6e13afac5978006d0e089bee3` |
| `depth_array_stride_gpu` | Depth32 buffer round trip using 768-byte image stride and scratch | `1d93f19e932c13c9c2b5c5041c285d3963ca2be78eec3ee3a7f37df4f5a02997` |
| `depth_array_cpu_gpu` | Depth32 CPU upload and texture copy/readback | `0556f1d275a98762d3d0ff35ea0ab9ed8e6a3b7869a321c083a0560bcd344402` |

Each array test uses distinct layer contents. A final candidate requires a new
run with retained full logs, source/commit identity, and independent review.
