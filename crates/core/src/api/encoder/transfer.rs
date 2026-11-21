//! Explicit copy, resolve, blit, and clear recording.
//!
//! The caller supplies resource states. Admission queries leave the encoder intact.

use super::*;

impl<B: CommandBackend> CommandEncoder<B> {
    pub fn copy_buffer(
        &mut self,
        src: &Buffer<B>,
        src_offset: u64,
        dst: &Buffer<B>,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<()> {
        if src.device != self.device || dst.device != self.device {
            return Err(Error::WrongDevice);
        }
        let result = self
            .call(|e| B::copy_buffer(e, &src.native, src_offset, &dst.native, dst_offset, bytes));
        if result.is_ok() {
            self.stats.record_copy();
        }
        result
    }
    pub fn copy_texture(
        &mut self,
        src: TextureRegion<'_, B>,
        dst: TextureRegion<'_, B>,
    ) -> Result<()> {
        self.validate_texture_transfer_regions(&src, &dst)?;
        let result = self.call(|e| B::copy_texture(e, src, dst));
        if result.is_ok() {
            self.stats.record_copy();
        }
        result
    }
    /// Checks whether the two regions can be copied on this encoder.
    /// The caller supplies the resource states required by the operation.
    pub fn copy_texture_support(
        &self,
        src: &TextureRegion<'_, B>,
        dst: &TextureRegion<'_, B>,
    ) -> Result<()> {
        self.validate_texture_transfer_regions(src, dst)?;
        if self.poisoned || self.in_pass {
            return Err(Error::InvalidState);
        }
        let native = self.native.as_ref().ok_or(Error::InvalidState)?;
        B::copy_texture_support(native, src, dst)
    }

    fn validate_texture_transfer_regions(
        &self,
        src: &TextureRegion<'_, B>,
        dst: &TextureRegion<'_, B>,
    ) -> Result<()> {
        if src.texture.device != self.device || dst.texture.device != self.device {
            return Err(Error::WrongDevice);
        }
        if src
            .texture
            .desc
            .as_ref()
            .is_some_and(|d| d.usage.contains(TextureUsage::TRANSIENT_ATTACHMENT))
            || dst
                .texture
                .desc
                .as_ref()
                .is_some_and(|d| d.usage.contains(TextureUsage::TRANSIENT_ATTACHMENT))
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }
    /// Checks a standalone resolve on this encoder without recording it.
    /// The caller supplies the resource states required by the operation.
    pub fn resolve_support(
        &self,
        src: &TextureRegion<'_, B>,
        dst: &TextureRegion<'_, B>,
    ) -> Result<()> {
        self.validate_texture_transfer_regions(src, dst)?;
        if self.poisoned || self.in_pass {
            return Err(Error::InvalidState);
        }
        let native = self.native.as_ref().ok_or(Error::InvalidState)?;
        B::resolve_support(native, src, dst)
    }
    pub fn resolve(&mut self, src: TextureRegion<'_, B>, dst: TextureRegion<'_, B>) -> Result<()> {
        self.validate_texture_transfer_regions(&src, &dst)?;
        let result = self.call(|e| B::resolve(e, src, dst));
        if result.is_ok() {
            self.stats.record_copy();
        }
        result
    }
    pub fn blit(
        &mut self,
        src: TextureRegion<'_, B>,
        dst: TextureRegion<'_, B>,
        filter: FilterMode,
    ) -> Result<()> {
        if src.texture.device != self.device || dst.texture.device != self.device {
            return Err(Error::WrongDevice);
        }
        if src
            .texture
            .desc
            .as_ref()
            .is_some_and(|d| d.usage.contains(TextureUsage::TRANSIENT_ATTACHMENT))
            || dst
                .texture
                .desc
                .as_ref()
                .is_some_and(|d| d.usage.contains(TextureUsage::TRANSIENT_ATTACHMENT))
        {
            return Err(Error::InvalidState);
        }
        let result = self.call(|e| B::blit(e, src, dst, filter));
        if result.is_ok() {
            self.stats.record_blit();
        }
        result
    }
    pub fn clear_buffer(&mut self, dst: &Buffer<B>, range: Range<u64>, value: u32) -> Result<()> {
        if dst.device != self.device {
            return Err(Error::WrongDevice);
        }
        let result = self.call(|e| B::clear_buffer(e, &dst.native, range, value));
        if result.is_ok() {
            self.stats.record_clear();
        }
        result
    }
    /// Clears all texels in each selected mip and array layer.
    /// The caller must arrange explicit transitions to the appropriate color
    /// attachment or depth-write state. Backends need not track or validate
    /// the actual current resource state.
    pub fn clear_texture(
        &mut self,
        dst: &Texture<B>,
        mips: Range<u32>,
        layers: Range<u32>,
        value: TextureClearValue,
    ) -> Result<()> {
        if dst.device != self.device {
            return Err(Error::WrongDevice);
        }
        if dst
            .desc
            .as_ref()
            .is_some_and(|d| d.usage.contains(TextureUsage::TRANSIENT_ATTACHMENT))
        {
            return Err(Error::InvalidState);
        }
        if mips.start >= mips.end || layers.start >= layers.end {
            return Err(Error::InvalidRange);
        }
        if let TextureClearValue::Depth(depth) = value {
            if !depth.is_finite() || !(0.0..=1.0).contains(&depth) {
                return Err(Error::InvalidRange);
            }
        }
        let result = self.call(|e| B::clear_texture(e, &dst.native, mips, layers, value));
        if result.is_ok() {
            self.stats.record_clear();
        }
        result
    }
    pub fn buffer_to_texture(
        &mut self,
        src: &Buffer<B>,
        layout: TexelCopyBufferLayout,
        dst: TextureRegion<'_, B>,
    ) -> Result<()> {
        self.validate_buffer_texture_copy(src, &dst)?;
        let result = self.call(|e| B::buffer_to_texture(e, &src.native, layout, dst));
        if result.is_ok() {
            self.stats.record_copy();
        }
        result
    }
    pub fn texture_to_buffer(
        &mut self,
        src: TextureRegion<'_, B>,
        dst: &Buffer<B>,
        layout: TexelCopyBufferLayout,
    ) -> Result<()> {
        self.validate_buffer_texture_copy(dst, &src)?;
        let result = self.call(|e| B::texture_to_buffer(e, src, &dst.native, layout));
        if result.is_ok() {
            self.stats.record_copy();
        }
        result
    }
    /// Checks the buffer layout and destination texture region without recording.
    /// The caller supplies the required resource states.
    pub fn buffer_to_texture_support(
        &self,
        src: &Buffer<B>,
        layout: TexelCopyBufferLayout,
        dst: &TextureRegion<'_, B>,
    ) -> Result<()> {
        self.validate_buffer_texture_copy(src, dst)?;
        if self.poisoned || self.in_pass {
            return Err(Error::InvalidState);
        }
        let native = self.native.as_ref().ok_or(Error::InvalidState)?;
        B::buffer_to_texture_support(native, &src.native, layout, dst)
    }
    /// Checks the source texture region and buffer layout without recording.
    /// The caller supplies the required resource states.
    pub fn texture_to_buffer_support(
        &self,
        src: &TextureRegion<'_, B>,
        dst: &Buffer<B>,
        layout: TexelCopyBufferLayout,
    ) -> Result<()> {
        self.validate_buffer_texture_copy(dst, src)?;
        if self.poisoned || self.in_pass {
            return Err(Error::InvalidState);
        }
        let native = self.native.as_ref().ok_or(Error::InvalidState)?;
        B::texture_to_buffer_support(native, src, &dst.native, layout)
    }
    fn validate_buffer_texture_copy(
        &self,
        buffer: &Buffer<B>,
        texture: &TextureRegion<'_, B>,
    ) -> Result<()> {
        if buffer.device != self.device || texture.texture.device != self.device {
            return Err(Error::WrongDevice);
        }
        if texture
            .texture
            .desc
            .as_ref()
            .is_some_and(|d| d.usage.contains(TextureUsage::TRANSIENT_ATTACHMENT))
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }
}
