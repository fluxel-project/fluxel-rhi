# Fluxel Naga patch

This directory is a local fork of crates.io `naga` 27.0.3, from the
[wgpu project](https://github.com/gfx-rs/wgpu). It retains Naga's upstream
`MIT OR Apache-2.0` license; see `LICENSE.MIT` and `LICENSE.APACHE`.
The upstream crates.io package checksum is
`066cf25f0e8b11ee0df221219010f213ad429855f57c494f995590c861a9a7d8`.
Upstream source layout is preserved; local regression tests live in the owning
DX12 crate rather than introducing a second test hierarchy in this copy.

## Scope

Relative to the upstream 27.0.3 release, Fluxel changes only:

- `src/back/hlsl/storage.rs`
- `src/back/hlsl/writer.rs`

The patch makes fixed storage-buffer binding arrays select their
`ByteAddressBuffer` descriptor separately from the byte offset within that
buffer. It covers HLSL load, store, and atomic emission. It also emits fixed
uniform struct binding arrays as typed `ConstantBuffer<Struct>` descriptor
arrays rather than invalid arrays of `cbuffer` declarations. Unsupported
dynamic/nested buffer binding arrays and non-struct uniform element types
return an HLSL writer error.

## Integration

Only Fluxel's DX12 crate selects this fork through a direct path dependency.
Vulkan, GL, and other crates continue to use their registry Naga dependency.

## Limits and removal

The patch supports fixed storage and uniform descriptor selection. Fluxel limits
its WGSL route to matching fixed, fully-bound compute layouts with explicit
non-uniform admission and no buffer-length queries. It
does not add runtime-sized or partially bound arrays, dynamic offsets for
arrays, nested binding arrays, or non-struct uniform-buffer array lowering.

Remove this fork when the selected upstream Naga release provides equivalent
fixed storage-buffer and uniform binding-array HLSL lowering, including
descriptor selection distinct from byte offsets where applicable, and Fluxel's
DX12 HLSL-output and GPU readback regressions pass unchanged against that
release.
