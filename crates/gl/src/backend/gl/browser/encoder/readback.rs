//! Immediate WebGL2 texture-to-buffer and buffer readback verbs.
//!
//! WebGL2 has no texture-to-buffer copy command.  The owner therefore reads
//! the RGBA8 source through the existing temporary-framebuffer `readPixels`
//! route, lays those top-left rows into the caller-declared padded layout, and
//! immediately uploads that byte range to the destination buffer.  This is a
//! direct execution fallback, not a recorded CPU copy or a later replay.

use super::super::owner::{BrowserBuffer, BrowserTexture};
use super::BrowserEncoder;
use fluxel_rhi_core::api::{Error, Result};

impl BrowserEncoder {
    /// Reads one RGBA8 texture rectangle and immediately stores it in a Core
    /// readback buffer using the caller's declared row pitch.
    pub(crate) fn copy_rgba8_texture_to_buffer(
        &mut self,
        source: &BrowserTexture,
        mip: u32,
        origin: [u32; 2],
        extent: [u32; 2],
        destination: &BrowserBuffer,
        destination_offset: u64,
        bytes_per_row: u32,
        rows_per_image: u32,
    ) -> Result<()> {
        self.same_owner(&source.owner)?;
        self.same_owner(&destination.owner)?;
        if extent.contains(&0) || rows_per_image < extent[1] {
            return Err(Error::InvalidRange);
        }
        let packed_row = extent[0].checked_mul(4).ok_or(Error::InvalidRange)?;
        if bytes_per_row < packed_row {
            return Err(Error::InvalidRange);
        }
        let body_len = u64::from(extent[1] - 1)
            .checked_mul(u64::from(bytes_per_row))
            .and_then(|prefix| prefix.checked_add(u64::from(packed_row)))
            .ok_or(Error::InvalidRange)?;
        let destination_end = destination_offset
            .checked_add(body_len)
            .ok_or(Error::InvalidRange)?;
        if destination_end > destination.size {
            return Err(Error::InvalidRange);
        }

        let tight = self.read_rgba8_texture(source, mip, origin, extent)?;
        let mut staged = vec![0; usize::try_from(body_len).map_err(|_| Error::OutOfMemory)?];
        let packed_row = usize::try_from(packed_row).map_err(|_| Error::OutOfMemory)?;
        let row_pitch = usize::try_from(bytes_per_row).map_err(|_| Error::OutOfMemory)?;
        for row in 0..usize::try_from(extent[1]).map_err(|_| Error::OutOfMemory)? {
            let source_start = row.checked_mul(packed_row).ok_or(Error::OutOfMemory)?;
            let destination_start = row.checked_mul(row_pitch).ok_or(Error::OutOfMemory)?;
            staged[destination_start..destination_start + packed_row]
                .copy_from_slice(&tight[source_start..source_start + packed_row]);
        }
        self.write_buffer(&destination.raw, destination_offset, &staged)
    }
}
