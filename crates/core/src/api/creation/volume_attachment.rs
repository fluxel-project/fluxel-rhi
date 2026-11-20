//! Single-depth-slice render attachments drawn from a 3D texture mip.

use super::*;

impl<B: ResourceBackend> Device<B> {
    /// Checks a color attachment view of one depth slice in a 3D texture mip.
    pub fn texture_3d_attachment_slice_support(
        &self,
        texture: &Texture<B>,
        format: TextureFormat,
        mip_level: u32,
        depth_slice: u32,
    ) -> Result<()> {
        let desc = self.volume_attachment_desc(texture, format, mip_level)?;
        B::texture_3d_attachment_slice_support(&self.native, &texture.native, &desc, depth_slice)
    }

    /// Creates an attachment-only view. Barriers still address the entire mip.
    pub fn create_texture_3d_attachment_slice(
        &self,
        texture: &Texture<B>,
        format: TextureFormat,
        mip_level: u32,
        depth_slice: u32,
    ) -> Result<TextureView<B>> {
        let desc = self.volume_attachment_desc(texture, format, mip_level)?;
        let native = B::create_texture_3d_attachment_slice(
            &self.native,
            &texture.native,
            &desc,
            depth_slice,
        )?;
        let mut view = self.wrap_view(texture, native);
        view.usage = Some(TextureUsage::RENDER_ATTACHMENT);
        Ok(view)
    }

    fn volume_attachment_desc(
        &self,
        texture: &Texture<B>,
        format: TextureFormat,
        mip_level: u32,
    ) -> Result<TextureViewDesc> {
        let end = mip_level.checked_add(1).ok_or(Error::InvalidRange)?;
        let desc = TextureViewDesc {
            format,
            dimension: TextureViewDimension::D3,
            aspects: TextureAspects::COLOR,
            mips: mip_level..end,
            layers: 0..1,
        };
        self.validate_view_request(texture, &desc)?;
        if !self.caps.texture_3d_attachment_slice {
            return Err(Error::Unsupported("3D texture attachment slice"));
        }
        Ok(desc)
    }
}
