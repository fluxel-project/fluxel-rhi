# ADR-0017: Keep presentation frames separate from textures and completion

**Status:** Accepted

## Context

Swapchain images, WebGPU current textures, CAMetalDrawables, and GL default
framebuffers do not share normal texture ownership. Present timing also differs
from GPU completion.

## Decision

`SurfaceImage` is distinct from `Texture` and `TextureView`; an acquired image
is non-cloneable and has one lifecycle. `PresentBinding` transfers its exclusive
lease into a submission plan. A rejected handoff can return the lease; an
unhanded lease is abandoned through its native drop path. Native present
acceptance, GPU completion, and present outcome remain separate observations.

## Consequences

Backend lowering can normalize attachments privately without leaking a
swapchain model. Backends own the safety of their native lease abandonment and
configuration-generation checks.
