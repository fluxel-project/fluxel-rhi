# Publishing the Fluxel RHI crates

All packages in `crates/` are intended to become independent crates.io packages. This guide records package ordering and the remaining registry-compatibility work. It does not authorize publishing or releasing a version.

## Package order

1. `fluxel-rhi-core` has no workspace package dependency and can be packaged independently.
2. Backend packages (`fluxel-rhi-dx12`, `fluxel-rhi-vulkan`, `fluxel-rhi-webgpu`, `fluxel-rhi-gl`, and `fluxel-rhi-metal`) depend on Core. Publish Core first, then publish backends as their registry dependencies become available.
3. `fluxel-rhi` is the facade. Publish it after every backend selected by its feature set is available from crates.io.

All current workspace package versions are `0.16.0`. Internal path dependencies declare the corresponding version so the same manifest can resolve packages from the registry after publication.

## Release blockers

### Git-only `fluxel-host` dependency

The facade has target-specific regular dependencies on `fluxel-host` from a Git URL. Several backend crates also use the Git dependency in development dependencies. Cargo packages cannot rely on a Git dependency for a crates.io release. Before publishing affected packages, publish a compatible `fluxel-host` version and switch these dependencies to registry versions, or remove/refactor the dependency while preserving the intended integration contract.

### DX12's local Naga patch

The DX12 package currently uses a local patched Naga source at version `27.0.3`. The local patch is not included as a replacement registry source when this package is published, so the package would resolve the registry Naga release instead. Confirm that the required behavior exists upstream; otherwise publish a properly versioned patched fork under a distinct package identity and depend on it. Validate the exact packaged dependency graph before release.

## Package contents and metadata

Each package has its own user-facing `README.md`, `LICENSE-APACHE`, and `LICENSE-MIT`. Architecture and design rationale are in that package's `DESIGN.md`; those files are repository documentation and are not required to build the package. Package manifests provide docs.rs metadata and point `readme` to the package README.

The root [README](README.md) describes the workspace. The [verification guide](documents/verification.md) describes validation procedures. The root [TODO](TODO.md) tracks cross-package deferred work. Package design links are collected in the READMEs and the [RHI design overview](crates/rhi/DESIGN.md).

## Pre-release checks

For each package, inspect the exact archive contents with `cargo package --list -p <package>`. Then run `cargo package -p <package>` or `cargo publish --dry-run -p <package>` after registry dependencies and the intended dependency graph are ready. Resolve packaging warnings and verify the generated archive contains its README, license files, source, and any package-owned assets. Finally validate documentation builds on the targets declared in that package's `[package.metadata.docs.rs]` configuration.

Do not publish from a dirty or unreviewed candidate. Pin the final validation evidence to the exact release commit and publish packages in dependency order.
