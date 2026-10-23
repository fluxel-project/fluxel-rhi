# Capability contracts

This file groups related decision records to keep the index compact. Original ADR identifiers and status labels are retained; current normative contracts live in the linked design and testing documents.

## ADR-0016: Publish capabilities only with lowering closure

**Status:** Accepted

## Context

Native API names, extension strings, traits, and partial lowering do not prove
that a portable operation is executable. Over-reporting moves predictable
errors from validation into submission or drivers.

## Decision

Capability facts are instance data. A `Supported` answer requires exact
validation, native lowering, resource retention, completion/loss behaviour,
and at least one conformance case. Incomplete paths stay absent and fail
structured before native work.

## Consequences

Facts may be asymmetric across formats, dimensions, stages, routes, and
platforms. Backend-private optimizations do not become public facts merely
because a native mechanism exists.

## ADR-0019: Define statistics as logical observation, not profiling

**Status:** Future design principle. The current Core API has no portable
statistics interface; this record does not claim an implemented capability.

## Context

Native counters have different availability, cost, and meaning. Treating them
as portable statistics would silently add work or report incomparable values.

## Decision

RHI statistics report logical object, command, submission, and lifecycle
observations. They never insert GPU commands or claim native profiling facts.
Native profiling remains an optional backend/tooling concern with its own
capability and conversion closure.

## Consequences

Statistics are predictable across backends and safe to query in diagnostics;
their values do not imply timing, pipeline counters, or driver metrics.

## ADR-0020: Admit optional feature families as complete portable contracts

**Status:** Accepted

## Context

Avoiding a feature solely because one baseline backend lacks it makes the API a
least-common-denominator wrapper. Publishing a feature from an enum alone is
equally unsafe.

## Decision

Each optional family has a portable vocabulary, precise capability facts or
queries, requirements negotiation, descriptor/recording validation, lowering,
retention, loss behaviour, and conformance evidence. Families include queries,
indirect commands, compressed formats, multiview, storage access, and advanced
shader/pipeline work. Advanced but incomplete lowerings retain their vocabulary
while their facts stay disabled.

## Consequences

`Unsupported` is a correct result, not an API omission. Current TODO families
include physical aliasing, mesh/task and ray paths, cooperative matrices,
external interop, multiplanar routes, native debugging tools, HDR/timing, and
advanced descriptor indexing. A reachable path must never use a placeholder
success or panic.
