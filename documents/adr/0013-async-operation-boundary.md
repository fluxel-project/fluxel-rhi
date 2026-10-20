# ADR-0013: Make only future-event operations asynchronous

**Status:** Accepted

## Context

Thread safety and a future event are different properties. Turning ordinary
logical creation and recording into futures obscures ownership without adding
an asynchronous boundary.

## Decision

Adapter/device request, shader/pipeline compilation, surface configuration and
acquisition, buffer mapping, completion observation, and idle waiting use
futures. These operations can depend on an externally completed event.

Queue submission is synchronous: its result describes native acceptance,
including partial or uncertain acceptance, rather than GPU completion. Command
recording and immediate execution are also synchronous. Presentation executes
through the deferred or immediate execution contract; it is not universally an
async operation and does not establish GPU completion.

Readback copy encoding is synchronous. Waiting for its completion and mapping
the resulting buffer are separate future-event operations. Capability queries,
logical buffer/texture/view/sampler/binding creation, validation, statistics,
and diagnostics remain synchronous.

## Consequences

Backends may use immediate-ready implementations for an async operation, while
concurrent synchronous calls remain ordinary calls. A new async API requires a
real externally completed event.
