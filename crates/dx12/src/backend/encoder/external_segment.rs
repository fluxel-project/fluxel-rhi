//! Exact native queue boundaries for external D3D12 synchronization.
//!
//! D3D12 fences live on command queues, whereas Fluxel commands live in a
//! command list. At an explicit acquire or release this module closes the
//! current list and starts another one, so submission can execute the native
//! `Wait` or `Signal` between them in the exact recorded order. It performs no
//! graph scheduling, hazard tracking, or command replay.

pub(crate) use super::core::QueueBoundary;
