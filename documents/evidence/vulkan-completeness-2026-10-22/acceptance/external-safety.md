# External import SPI safety follow-up

Source: `76657d278e8bb0250a66e12ac43d53ce612577ce`.

The external import SPI now requires unsafe calls, matching Core's existing
unsafe import boundary. Native-handle validity, allocation metadata,
synchronization and independent host-mapping aliases are caller obligations.
Safe callers can no longer bypass these obligations through the public SPI.
The Vulkan implementation changes signatures only; resource execution is
unchanged.

Rust 1.87 Core doctests passed both compile-fail cases. The Vulkan external
buffer and texture hardware fixtures each passed on the previously recorded
AMD Radeon 780M with `VK_LAYER_KHRONOS_validation`; their captured output
contains no VUID or validation error. Commands, raw bytes and source status
are in [logs](logs/external-safety/). Metal work was concurrently uncommitted;
the tested Core and Vulkan source directories matched this commit.

This targeted follow-up does not claim that the earlier full workspace or
23-example matrix was repeated.
