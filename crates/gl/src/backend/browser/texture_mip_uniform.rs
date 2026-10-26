//! WebGL uniform update for the sampled texture's exposed mip count.
use super::owner::{BrowserEncoder, BrowserRasterProgram};
use fluxel_rhi_core::api::{Error, Result};

impl BrowserEncoder {
    /// Sets a compiler-generated uint uniform in the current raster program.
    /// An optimized-out uniform has no location and requires no native write.
    pub(crate) fn set_texture_mip_count(
        &mut self,
        program: &BrowserRasterProgram,
        name: &str,
        count: u32,
    ) -> Result<()> {
        self.same_owner(&program.owner)?;
        self.require_raster()?;
        if count == 0 {
            return Err(Error::InvalidRange);
        }
        let location = program.uniform_location(name)?;
        self.with_context("set texture mip count", |raw| {
            if let Some(location) = location.as_ref() {
                raw.uniform1ui(Some(&location), count);
            }
        })
    }
}
