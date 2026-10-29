# Publishing the Fluxel RHI crates

This release publishes the seven packages in `crates/` as `0.1.0`, plus two registry dependencies required by their examples and backend behavior: `fluxel-host` and `fluxel-naga`. Publishing is permanent for each package version, so validate each archive before upload.

## Package order

1. **`fluxel-host` 0.1.0** — independent native-window and lifecycle package from the sibling repository. RHI tests and examples use it; RHI library code does not require it at runtime.
2. **`fluxel-rhi-core` 0.1.0** — portable API and backend contract.
3. **`fluxel-naga` 0.1.0** — Fluxel's narrow Naga fork used by DX12 for fixed HLSL storage/uniform descriptor-array lowering. Its Rust library name remains `naga`.
4. **Backend crates** — `fluxel-rhi-dx12`, `fluxel-rhi-vulkan`, `fluxel-rhi-webgpu`, `fluxel-rhi-gl`, and `fluxel-rhi-metal`, all `0.1.0` and dependent on Core. DX12 also depends on `fluxel-naga`.
5. **`fluxel-rhi` 0.1.0** — facade after every backend package exists on crates.io.

The seven RHI packages share the `fluxel-rhi` repository; `fluxel-host` lives at `https://github.com/fluxel-project/fluxel-host`. Each package manifest must retain its own one-sentence `description`, `repository`, `readme`, license expression, and docs.rs link. The fork's repository is `https://github.com/fluxel-project/fluxel-rhi` and its README describes its upstream provenance and patch scope.

## Registry dependency rules

- Local path dependencies between RHI packages include an exact compatible version. Cargo packages replace local paths with registry requirements.
- `fluxel-host` appears only in development dependencies (examples and tests) for RHI packages. Those manifests pin the host Git commit for local tests/examples and also specify version `=0.1.0`; packaging uses the registry version. The host crate itself has no Git or path dependencies.
- DX12 depends on `fluxel-naga` through the vendored source path plus version `=0.1.0`. Packaging rewrites that path to the registry package. Do not replace it with upstream `naga` 27.0.3 unless equivalent fixed descriptor-array HLSL lowering is confirmed and the DX12 shader regressions pass unchanged.

## Package checks and release sequence

Before publishing, run `cargo package --list` for every package and inspect the archive for its README, both license files, Rust source, build scripts, shader/compiler assets, and fixtures needed by package verification. Run package verification for host and `fluxel-naga`, then `cargo package`/`cargo publish --dry-run` for each RHI package after its registry dependencies are available. Run the workspace's targeted compile/test gates and the applicable platform checks before freezing the release commit.

Commit and push the final candidate for each repository, create an annotated `crates-v0.1.0` tag when the historical `v0.1.0` tag already points at another commit (otherwise use `v0.1.0`), and confirm the remote tag points at that commit. Publish in dependency order with `cargo publish`; after each upload, verify the version through crates.io and proceed only when Cargo can resolve it. Do not publish from a dirty or unreviewed candidate. No GitHub Release or release attachment is part of this process.
