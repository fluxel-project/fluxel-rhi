# `fluxel-rhi-core`

`fluxel-rhi-core` defines the backend-independent GPU API used by Fluxel
renderers and hosts. It provides typed resources, device and surface contracts,
capability queries, execution plans, synchronization, and the implementation
SPI used by backend crates. It does not create a GPU device or include a native
backend.

## Add the dependency

```toml
[dependencies]
fluxel-rhi-core = "0.1.0"
```

Application code will usually use the [`fluxel-rhi` facade](../rhi/README.md),
which reexports this API and selects a concrete backend through Cargo features.
Depend on Core directly when implementing a backend or writing generic code
against the shared contract.

```rust
use fluxel_rhi_core::api;
```

Start with the [`api` module](https://docs.rs/fluxel-rhi-core/latest/fluxel_rhi_core/api/).
The public contract is statically typed by backend; it does not provide runtime
backend switching.

## Design and compatibility

See [`DESIGN.md`](DESIGN.md) for Core's ownership and execution boundaries.
Callers provide ordered work; RenderGraph planning, hazard analysis, and
resource retirement remain in higher layers. Unsupported operations return
explicit capability or validation errors.

The crate requires Rust 1.87 or newer and uses the `MIT OR Apache-2.0` license.
See the repository [verification guide](https://github.com/fluxel-project/fluxel-rhi/blob/main/documents/verification.md)
for backend conformance scope.

Repository: https://github.com/fluxel-project/fluxel-rhi
