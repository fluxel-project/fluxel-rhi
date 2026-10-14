# ADR-0023: Keep host windows separate from RHI presentation integration

**Status:** Accepted

## Context

The host owns native window and canvas lifetimes and reports lifecycle events.
The RHI owns surface configuration and the exclusive acquired-image lease.
A copied `HWND`, `ANativeWindow*`, `UIView*`, or browser canvas handle cannot
extend the host object's lifetime.

## Decision

Keep the host and RHI ownership domains distinct. Platform integration code
borrows the live host handle to create a backend `Surface`, and keeps the host
object alive until that surface and all its `SurfaceImage` leases retire.
Backend-specific native handles and surface registration stay behind the
backend boundary. Core exposes `Surface`, `SurfaceImage`, and `PresentBinding`
without depending on a host or browser session type.

The current Windows examples put this integration in `examples/utils`; the
DX12 and Vulkan constructors explicitly mark raw-window-handle lifetime as an
unsafe caller obligation. WebGPU accepts a browser canvas through its backend
constructor. These routes are platform integration code, not another RHI
execution domain, resource manager, or RenderGraph.

On host destruction, integration code must retire the surface and its image
leases before releasing the native host object. A stale acquired-image
generation must reject present handoff through the surface contract.

## Consequences

- A platform route must document how it keeps the host target alive through
  surface use and how resize or destruction invalidates outstanding leases.
- Native host setup belongs in the platform integration or test fixture; the
  portable workload uses Core RHI operations once it has a surface.
- Backends can use different native presentation objects without adding host
  window types to the Core API.
