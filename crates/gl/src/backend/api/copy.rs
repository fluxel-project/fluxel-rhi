//! Copy, upload, and readback contracts with reversible pixel-store state.

use super::{GlBufferRange, GlError, GlFamilyApi, GlTextureRegion};

/// The byte encoding supplied to or returned from a pixel transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlPixelFormat {
    Rgba8,
    Bgra8,
    Rgb8,
    Red8,
    Rg8,
    /// Uncompressed texels whose layout uses this many bytes.
    TexelBytes(u8),
    Stencil8,
    Depth16,
    Depth24Stencil8,
    Depth32Float,
    /// Compressed data is laid out in complete encoded blocks.
    Compressed {
        block_width: u8,
        block_height: u8,
        block_bytes: u8,
    },
}

impl GlPixelFormat {
    pub(crate) const fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Rgba8 | Self::Bgra8 | Self::Depth24Stencil8 | Self::Depth32Float => 4,
            Self::Rgb8 => 3,
            Self::Red8 | Self::Stencil8 => 1,
            Self::Depth16 | Self::Rg8 => 2,
            Self::TexelBytes(bytes) => bytes as u32,
            Self::Compressed { block_bytes, .. } => block_bytes as u32,
        }
    }

    pub(crate) const fn block_extent(self) -> (u32, u32) {
        match self {
            Self::Compressed {
                block_width,
                block_height,
                ..
            } => (block_width as u32, block_height as u32),
            _ => (1, 1),
        }
    }
}

/// Whether a row stride which cannot be expressed by GL pixel-store is repacked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlRepackPolicy {
    Disallow,
    /// Allows an implementation-owned temporary staging allocation up to this exact bound.
    Bounded {
        max_bytes: u64,
    },
}

/// Explicit client-memory row layout. `bytes_per_row` includes caller-owned padding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlPixelLayout {
    pub format: GlPixelFormat,
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
    pub offset: u64,
    pub alignment: u8,
    pub repack: GlRepackPolicy,
}
impl GlPixelLayout {
    /// Whether GL's own pixel-store can express this layout's row stride.
    ///
    /// One question, asked by the validator, by both staging directions and by
    /// both pixel-store halves.  They have to answer it the same way, because
    /// the answer is what decides whether GL is describing the caller's buffer
    /// or a tightly packed staging one.
    pub(crate) fn stride_is_mappable(self) -> bool {
        self.bytes_per_row
            .is_multiple_of(self.format.bytes_per_pixel())
            && self.bytes_per_row.is_multiple_of(u32::from(self.alignment))
    }

    /// The `(ALIGNMENT, ROW_LENGTH)` pair GL is given for this layout.
    ///
    /// The mappable arm describes the caller's own buffer, which is the thing
    /// the payload is sliced from.  The other arm describes tightly staged
    /// rows: row length zero means "the sub-image's own width" and alignment
    /// one means nothing pads it.
    ///
    /// `PACK_*` and `UNPACK_*` take the same numbers for the same layout, so
    /// both directions read them from here instead of deciding again.  They had
    /// been decided twice, and the unpack half had drifted: it described the
    /// caller's pitch over a staging buffer that did not have that pitch, which
    /// is an out-of-bounds read on the native provider and a structured error
    /// on the browser one.
    pub(crate) fn staged_row_parameters(self) -> (u32, u32) {
        if self.stride_is_mappable() {
            (
                u32::from(self.alignment),
                self.bytes_per_row / self.format.bytes_per_pixel(),
            )
        } else {
            (1, 0)
        }
    }

    pub(crate) fn validate_for(self, region: GlTextureRegion) -> Result<(), GlCopyValidationError> {
        // First, because `required_bytes_unchecked` subtracts one from the
        // height and from the layer count: a zero extent would underflow there
        // -- panicking in debug, wrapping to `u32::MAX` in release -- and be
        // reported as a size problem rather than as the shape problem it is.
        // The guard has to live in this validator, because this is the one the
        // providers call; the `validate_for_basic` that carried it was
        // reachable from nowhere.
        if region.extent.width == 0
            || region.extent.height == 0
            || region.extent.depth_or_layers == 0
        {
            return Err(GlCopyValidationError::ZeroTransferExtent);
        }
        let (block_width, block_height) = self.format.block_extent();
        let packed = match region
            .extent
            .width
            .div_ceil(block_width)
            .checked_mul(self.format.bytes_per_pixel())
        {
            Some(v) => v,
            None => return Err(GlCopyValidationError::LayoutOverflow),
        };
        if self.bytes_per_row < packed {
            return Err(GlCopyValidationError::RowTooShort);
        }
        if self.rows_per_image < region.extent.height.div_ceil(block_height) {
            return Err(GlCopyValidationError::ImageTooShort);
        }
        if !matches!(self.alignment, 1 | 2 | 4 | 8) {
            return Err(GlCopyValidationError::InvalidPixelAlignment);
        }
        if !self.stride_is_mappable() {
            match self.repack {
                GlRepackPolicy::Disallow => return Err(GlCopyValidationError::UnmappableRowStride),
                GlRepackPolicy::Bounded { max_bytes }
                    if self.required_bytes_unchecked(region)? > max_bytes =>
                {
                    return Err(GlCopyValidationError::RepackBoundExceeded);
                }
                GlRepackPolicy::Bounded { .. } => {}
            }
        }
        Ok(())
    }
    pub(crate) fn required_bytes(
        self,
        region: GlTextureRegion,
    ) -> Result<u64, GlCopyValidationError> {
        self.validate_for(region)?;
        self.required_bytes_unchecked(region)
    }
    fn required_bytes_unchecked(
        self,
        region: GlTextureRegion,
    ) -> Result<u64, GlCopyValidationError> {
        let images = u64::from(region.extent.depth_or_layers - 1);
        let image_stride = u64::from(self.bytes_per_row)
            .checked_mul(u64::from(self.rows_per_image))
            .ok_or(GlCopyValidationError::LayoutOverflow)?;
        let (block_width, block_height) = self.format.block_extent();
        let final_rows = u64::from(region.extent.height.div_ceil(block_height) - 1)
            .checked_mul(u64::from(self.bytes_per_row))
            .ok_or(GlCopyValidationError::LayoutOverflow)?;
        let final_width = u64::from(region.extent.width.div_ceil(block_width))
            .checked_mul(u64::from(self.format.bytes_per_pixel()))
            .ok_or(GlCopyValidationError::LayoutOverflow)?;
        self.offset
            .checked_add(
                images
                    .checked_mul(image_stride)
                    .ok_or(GlCopyValidationError::LayoutOverflow)?,
            )
            .and_then(|v| v.checked_add(final_rows))
            .and_then(|v| v.checked_add(final_width))
            .ok_or(GlCopyValidationError::LayoutOverflow)
    }
}

/// Complete GL pixel-store state that a scoped transfer must restore, even after driver error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlPixelStoreState {
    pub pack_alignment: u8,
    pub unpack_alignment: u8,
    pub pack_row_length: u32,
    pub unpack_row_length: u32,
    pub pack_skip_pixels: u32,
    pub pack_skip_rows: u32,
    pub pack_image_height: u32,
    pub pack_skip_images: u32,
    pub unpack_skip_pixels: u32,
    pub unpack_skip_rows: u32,
    pub unpack_image_height: u32,
    pub unpack_skip_images: u32,
}
impl GlPixelStoreState {
    pub(crate) const DEFAULT: Self = Self {
        pack_alignment: 4,
        unpack_alignment: 4,
        pack_row_length: 0,
        unpack_row_length: 0,
        pack_skip_pixels: 0,
        pack_skip_rows: 0,
        pack_image_height: 0,
        pack_skip_images: 0,
        unpack_skip_pixels: 0,
        unpack_skip_rows: 0,
        unpack_image_height: 0,
        unpack_skip_images: 0,
    };
    pub(crate) fn validate(self) -> Result<(), GlCopyValidationError> {
        if !matches!(self.pack_alignment, 1 | 2 | 4 | 8)
            || !matches!(self.unpack_alignment, 1 | 2 | 4 | 8)
        {
            return Err(GlCopyValidationError::InvalidPixelAlignment);
        }
        Ok(())
    }
    /// Validates image-height settings against a concrete transfer before mutation.
    pub(crate) fn validate_for_transfer(
        self,
        region: GlTextureRegion,
    ) -> Result<(), GlCopyValidationError> {
        self.validate()?;
        if (self.pack_image_height != 0 && self.pack_image_height < region.extent.height)
            || (self.unpack_image_height != 0 && self.unpack_image_height < region.extent.height)
        {
            return Err(GlCopyValidationError::ImageHeightTooShort);
        }
        Ok(())
    }
}

/// A successful readback always identifies the layout used for the returned bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GlReadback {
    pub layout: GlPixelLayout,
    pub bytes: Vec<u8>,
}
impl GlReadback {
    /// Ensures a successful provider result is neither truncated nor silently overlong.
    pub(crate) fn validate_for(
        &self,
        source: GlTextureRegion,
    ) -> Result<(), GlCopyValidationError> {
        let required = self.layout.required_bytes(source)?;
        if u64::try_from(self.bytes.len()).ok() != Some(required) {
            return Err(GlCopyValidationError::ReadbackLengthMismatch);
        }
        Ok(())
    }
}

/// Copy validation failures which must occur before a GL command or pixel-store mutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlCopyValidationError {
    RowTooShort,
    ImageTooShort,
    LayoutOverflow,
    InvalidPixelAlignment,
    UploadSourceTooShort,
    UnmappableRowStride,
    RepackBoundExceeded,
    ReadbackLengthMismatch,
    ZeroTransferExtent,
    ImageHeightTooShort,
}

/// Copy operations. Pixel-store changes are scoped: implementations snapshot `pixel_store`,
/// apply a transfer layout, and restore the exact snapshot before every return path.
///
/// Buffer binding points are deliberately *not* scoped. `COPY_READ_BUFFER` and
/// `COPY_WRITE_BUFFER` are private to the transfer implementation: a provider
/// binds the buffer it is about to use immediately before the command and
/// leaves the target bound on return. Nothing needs restoring because no verb
/// in this layer accepts a caller-supplied target, so neither the Layer 2 state
/// mirror nor a caller can hold an opinion about these targets — the mirror
/// covers only binding points some verb can name. Adding a restore here would
/// imply the opposite contract; adding a target-taking verb would break it.
/// `scripts/check_gl_architecture.py` confines the constants to provider
/// execution bodies so the two cannot drift apart.
pub(crate) trait GlCopyDomainApi: GlFamilyApi {
    fn buffer_to_texture(
        &mut self,
        _: GlBufferRange,
        _: GlTextureRegion,
        _: GlPixelLayout,
    ) -> Result<(), GlError> {
        Err(GlError::Unsupported {
            operation: "buffer-to-texture",
            reason: "provider has no pixel buffer upload",
        })
    }
    fn copy_buffer_range(
        &mut self,
        source: GlBufferRange,
        destination: GlBufferRange,
    ) -> Result<(), GlError>;
    fn copy_texture_region(
        &mut self,
        source: GlTextureRegion,
        destination: GlTextureRegion,
    ) -> Result<(), GlError>;
    /// Uploads exactly `range.size` bytes at `range.offset` in one live buffer.
    fn upload_buffer(&mut self, destination: GlBufferRange, bytes: &[u8]) -> Result<(), GlError>;
    /// Reads back exactly `range.size` bytes from one live buffer.
    fn read_buffer(&mut self, source: GlBufferRange) -> Result<Vec<u8>, GlError>;
    fn upload_texture(
        &mut self,
        destination: GlTextureRegion,
        layout: GlPixelLayout,
        bytes: &[u8],
    ) -> Result<(), GlError>;
    fn read_texture(
        &mut self,
        source: GlTextureRegion,
        layout: GlPixelLayout,
    ) -> Result<GlReadback, GlError>;
    fn pixel_store(&self) -> GlPixelStoreState;
}

#[cfg(test)]
#[path = "../../tests/backend/api/copy__tests.rs"]
mod tests;
