# `fluxel-naga`

`fluxel-naga` is a Fluxel-maintained build of Naga 27.0.3 with narrowly scoped HLSL fixes for fixed storage-buffer and uniform-buffer binding arrays. It preserves the upstream Rust library name `naga`; DX12 consumes it as a drop-in compiler dependency.

Repository: https://github.com/fluxel-project/fluxel-rhi

## Why this fork exists

The upstream 27.0.3 HLSL writer does not correctly select fixed storage-buffer array descriptors independently from byte offsets and cannot emit the fixed uniform-struct descriptor-array form required by Fluxel's DX12 backend. This fork corrects those routes. Unsupported dynamic or nested buffer arrays still return a writer error.

Only `fluxel-rhi-dx12` uses this fork. Other Fluxel backends continue to use upstream `naga` from crates.io. The fork will be removed when an upstream release provides equivalent lowering and the DX12 regressions pass against it.

## Compatibility and license

The package is named `fluxel-naga`, while its Rust library crate remains `naga`. It is based on upstream Naga 27.0.3 and retains the upstream `MIT OR Apache-2.0` license. See [`FLUXEL-PATCH.md`](FLUXEL-PATCH.md) for the patch scope and provenance; upstream source licenses are included in this package.

Repository: https://github.com/fluxel-project/fluxel-rhi
