//! Owned handle for one backend-imported external video-frame snapshot.
//!
//! The producer owns decoding, frame updates, color metadata and source
//! invalidation policy. Core only retains the backend import needed to bind
//! and submit this snapshot.
use super::*;

/// A single immutable video-frame snapshot imported by one RHI device.
pub struct VideoFrame<B: Backend> {
    pub(crate) device: DeviceId,
    pub(crate) native: B::NativeVideoFrame,
}

impl<B: VideoBackend> Device<B> {
    /// Imports one video-frame source through the backend's external-texture
    /// route. The source is consumed so the backend can retain it for the
    /// imported snapshot's native lifetime.
    pub fn import_video_frame(&self, source: B::NativeVideoSource) -> Result<VideoFrame<B>> {
        Ok(VideoFrame {
            device: self.id,
            native: B::import_video_frame(&self.native, source)?,
        })
    }
}

impl<B: Backend> VideoFrame<B> {
    /// Returns the device which imported this frame.
    pub fn device_id(&self) -> DeviceId {
        self.device
    }

    /// Borrows the backend import for its native binding path.
    ///
    /// # Safety
    /// The caller must have validated the frame's device identity before
    /// passing the native value to that device's backend.
    #[doc(hidden)]
    pub unsafe fn native_ref(&self) -> &B::NativeVideoFrame {
        &self.native
    }
}
