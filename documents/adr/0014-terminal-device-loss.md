# ADR-0014: Treat device loss as a terminal device identity state

**Status:** Accepted lifecycle principle; historical protocol target, not a
completed cross-backend Core contract.

The current authority is [the RHI design](../design-rhi.md) and
[`core::api`](../../crates/core/src/api/mod.rs). Core uses `DeviceId` and does
not expose generic `status()` or `loss_info()` methods. The protocol below
records the original target, not APIs or uniform wake-up guarantees already
implemented by every backend. WebGL2's implemented terminal-generation behavior
is described separately in the current design. A shared loss-query API requires
agreement on native loss authority, observation semantics, and pending-operation
wake-up behavior across backends before it can be claimed as supported.

## Context

Native APIs cannot revive a removed device safely, and browser contexts have
the same ownership problem. A mutable generation counter would permit stale
objects to appear valid after recovery.

## Decision

Loss is terminal for one `DeviceIdentity`. It wakes pending completion,
readback, acquire, present, and idle futures; later operations return structured
`DeviceLost`; `status()` and `loss_info()` remain synchronously queryable. A
new request creates a distinct identity. There is no required `wait_lost()` API.

## Consequences

Callers recover by rebuilding device-scoped objects. Backends must route every
native loss observation through the same terminal authority and never publish
readback success after loss.
