//! Optional raster features such as indirect draws and multiview.
//!
//! These operations are admitted only by backend capabilities. Their ordering,
//! resource state, and fallback policy remain outside Core.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]
use super::*;
use std::{
    future::Future,
    marker::PhantomData,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

impl<B: Backend> RasterExtrasOps<B> for Unavailable {
    const AVAILABLE: bool = false;
    fn draw_indirect(_: &mut B::NativeEncoder, _: &B::NativeBuffer, _: u64, _: bool) -> Result<()> {
        Err(Error::Unsupported("indirect"))
    }
    fn multi_draw(
        _: &mut B::NativeEncoder,
        _: &B::NativeBuffer,
        _: u64,
        _: u32,
        _: u32,
        _: bool,
    ) -> Result<()> {
        Err(Error::Unsupported("multidraw"))
    }
    fn multi_draw_count(
        _: &mut B::NativeEncoder,
        _: &B::NativeBuffer,
        _: u64,
        _: &B::NativeBuffer,
        _: u64,
        _: u32,
        _: u32,
        _: bool,
    ) -> Result<()> {
        Err(Error::Unsupported("multidraw-count"))
    }
    fn set_push_constants(_: &mut B::NativeEncoder, _: StageMask, _: u32, _: &[u8]) -> Result<()> {
        Err(Error::Unsupported("push constants"))
    }
}
pub struct RasterExtras<'a, B: Backend> {
    device: &'a Device<B>,
}
impl<B: Backend> Device<B> {
    pub fn raster_extras(&self) -> Option<RasterExtras<'_, B>> {
        (self.caps.indirect_draw
            || self.caps.multi_draw
            || self.caps.multi_draw_count
            || self.caps.push_constants)
            .then_some(RasterExtras { device: self })
    }
}
impl<B: Backend> RasterExtras<'_, B> {
    /// All indirect argument records in this domain must use a zero
    /// first-instance field unless `DeviceCapabilities::indirect_first_instance`
    /// is true. This includes GPU-generated records and count-based draws;
    /// backends do not inspect argument contents on the CPU.
    /// The caller must transition the argument buffer to `INDIRECT_READ` before the pass.
    pub fn draw_indirect(
        &self,
        pass: &mut RasterPass<'_, B>,
        args: &Buffer<B>,
        offset: u64,
        indexed: bool,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if !self.device.caps.indirect_draw {
            return Err(Error::Unsupported("indirect draw"));
        }
        if pass.encoder.device != self.device.id || args.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        if offset % 4 != 0 {
            return Err(Error::InvalidRange);
        }
        let result = pass.call(|e| B::Extras::draw_indirect(e, &args.native, offset, indexed));
        if result.is_ok() {
            if indexed {
                pass.encoder.stats.record_draw_indexed();
            } else {
                pass.encoder.stats.record_draw();
            }
        }
        result
    }
    /// The caller must transition the argument buffer to `INDIRECT_READ` before the pass.
    pub fn multi_draw(
        &self,
        pass: &mut RasterPass<'_, B>,
        args: &Buffer<B>,
        offset: u64,
        count: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if !self.device.caps.multi_draw {
            return Err(Error::Unsupported("multi draw"));
        }
        if pass.encoder.device != self.device.id || args.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        if offset % 4 != 0
            || (count != 0 && (stride < indirect_argument_size(indexed) || stride % 4 != 0))
        {
            // Validate the portable command-word layout before lowering: native
            // APIs otherwise disagree on whether malformed stride is ignored,
            // clamped, or diagnosed after partial command recording.
            return Err(Error::InvalidRange);
        }
        let result =
            pass.call(|e| B::Extras::multi_draw(e, &args.native, offset, count, stride, indexed));
        if result.is_ok() {
            if indexed {
                pass.encoder.stats.record_draw_indexed();
            } else {
                pass.encoder.stats.record_draw();
            }
        }
        result
    }
    /// Draws up to `max` commands using the GPU-written count. The caller must
    /// transition both argument and count buffers to `INDIRECT_READ` before the
    /// pass; this also applies when both ranges belong to the same buffer.
    pub fn multi_draw_count(
        &self,
        pass: &mut RasterPass<'_, B>,
        args: &Buffer<B>,
        args_offset: u64,
        count: &Buffer<B>,
        count_offset: u64,
        max: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if !self.device.caps.multi_draw_count {
            return Err(Error::Unsupported("multi draw count"));
        }
        if pass.encoder.device != self.device.id
            || args.device != self.device.id
            || count.device != self.device.id
        {
            return Err(Error::WrongDevice);
        }
        if args_offset % 4 != 0
            || count_offset % 4 != 0
            || (max != 0 && (stride < indirect_argument_size(indexed) || stride % 4 != 0))
        {
            return Err(Error::InvalidRange);
        }
        let result = pass.call(|e| {
            B::Extras::multi_draw_count(
                e,
                &args.native,
                args_offset,
                &count.native,
                count_offset,
                max,
                stride,
                indexed,
            )
        });
        if result.is_ok() {
            if indexed {
                pass.encoder.stats.record_draw_indexed();
            } else {
                pass.encoder.stats.record_draw();
            }
        }
        result
    }
    pub fn push_constants(
        &self,
        pass: &mut RasterPass<'_, B>,
        stages: StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if !self.device.caps.push_constants {
            return Err(Error::Unsupported("push constants"));
        }
        if pass.encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let result = pass.call(|e| B::Extras::set_push_constants(e, stages, offset, bytes));
        if result.is_ok() {
            pass.encoder.stats.record_push_constant_write();
        }
        result
    }
}
