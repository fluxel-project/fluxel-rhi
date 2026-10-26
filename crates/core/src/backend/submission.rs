//! Backend SPI for completion observation and deferred native submission.
//!
//! These traits are implemented only by backend crates. The renderer-facing
//! completion values, packets, and queue facade remain in [`crate::api`].

use crate::{
    api::{CompletionState, DeferredTail, LaneId, Result, SubmitOutcome, SubmitPacket},
    backend::{Backend, CommandBackend},
};
use std::future::Future;

/// Completion observation supplied by one backend implementation.
pub trait CompletionBackend: Backend {
    type WaitFuture<'a>: Future<Output = Result<()>> + 'a
    where
        Self: 'a;
    fn completion_status(
        device: &Self::NativeDevice,
        point: &Self::NativeCompletion,
    ) -> CompletionState;
    fn wait_completion<'a>(
        device: &'a Self::NativeDevice,
        point: &'a Self::NativeCompletion,
    ) -> Self::WaitFuture<'a>;
    fn wait_idle<'a>(device: &'a Self::NativeDevice) -> Self::WaitFuture<'a>;
}

/// Deferred-only SPI. GL does not implement it. Native RecordedWork is one-shot;
/// neither this API nor the backend promises capture, record, or replay.
pub trait DeferredBackend: CommandBackend + CompletionBackend {
    type NativeRecordedWork;
    fn finish_encoder(
        encoder: Self::NativeEncoder,
        lane: LaneId,
    ) -> Result<Self::NativeRecordedWork>;
    /// # Safety
    /// Caller retains referenced objects, supplies valid sync and lifetimes.
    /// Return a truthful accepted prefix. On partial acceptance, emergency
    /// frontier evidence is required even when completion was None.
    unsafe fn submit(
        device: &Self::NativeDevice,
        packet: SubmitPacket<Self>,
    ) -> SubmitOutcome<Self>;
    /// # Safety
    /// Caller has observed every accepted-frontier checkpoint complete and
    /// continues to retain all resources through true GPU completion.
    unsafe fn resume(device: &Self::NativeDevice, tail: DeferredTail<Self>) -> SubmitOutcome<Self>;
}
