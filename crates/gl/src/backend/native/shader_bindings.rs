//! Native immediate binding commands.

use super::provider::NativeGlProvider;
use crate::backend::api::{
    BufferId, GlBindingApi, GlBindingLimits, GlBufferUsage, GlError, GlFamilyApi as _,
    GlTextureTarget, SamplerId, TextureId, validate_texture_unit, validate_uniform_buffer_binding,
    validate_uniform_range,
};

impl GlBindingApi for NativeGlProvider {
    fn active_texture(&mut self, unit: u32) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "active-texture";
        self.assert_ready(OP)?;
        validate_texture_unit(unit, self.binding_limits())
            .map_err(|error| Self::validation(OP, error.message()))?;
        // SAFETY: current-context contract; the unit was bounds-checked.
        unsafe { self.gl.active_texture(glow::TEXTURE0 + unit) };
        self.driver_error(OP)
    }

    fn bind_texture(
        &mut self,
        unit: u32,
        target: GlTextureTarget,
        texture: Option<TextureId>,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-texture";
        self.assert_ready(OP)?;
        validate_texture_unit(unit, self.binding_limits())
            .map_err(|error| Self::validation(OP, error.message()))?;
        let raw = texture
            .map(|texture| {
                let entry = self.texture(OP, texture)?;
                Ok::<_, GlError>(entry.0)
            })
            .transpose()?;
        // SAFETY: current-context contract; liveness and unit bounds checked.
        unsafe {
            self.gl.active_texture(glow::TEXTURE0 + unit);
            self.gl.bind_texture(texture_target(target), raw);
        }
        self.driver_error(OP)?;
        let class = match texture {
            Some(id) => super::sampler_border::numeric_class(self.texture(OP, id)?.1.format),
            None => super::sampler_border::NumericClass::Float,
        };
        self.texture_unit_classes.insert(unit, class);
        if let Some(sampler) = self.texture_unit_samplers.get(&unit).copied() {
            self.bind_sampler_variant(unit, Some(sampler), class)?;
        }
        Ok(())
    }

    fn bind_sampler(&mut self, unit: u32, sampler: Option<SamplerId>) -> Result<(), GlError> {
        const OP: &str = "bind-sampler";
        self.assert_ready(OP)?;
        validate_texture_unit(unit, self.binding_limits())
            .map_err(|error| Self::validation(OP, error.message()))?;
        let class = self
            .texture_unit_classes
            .get(&unit)
            .copied()
            .unwrap_or(super::sampler_border::NumericClass::Float);
        self.bind_sampler_variant(unit, sampler, class)?;
        if let Some(id) = sampler {
            self.texture_unit_samplers.insert(unit, id);
        } else {
            self.texture_unit_samplers.remove(&unit);
        }
        Ok(())
    }

    fn bind_uniform_buffer(
        &mut self,
        index: u32,
        buffer: Option<BufferId>,
        offset: u64,
        size: u64,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-uniform-buffer";
        self.assert_ready(OP)?;
        let limits = self.binding_limits();
        validate_uniform_buffer_binding(index, buffer, offset, size, limits)
            .map_err(|error| Self::validation(OP, error.message()))?;
        let Some(buffer) = buffer else {
            // SAFETY: current-context contract; the unbind shape was checked.
            unsafe {
                self.gl.bind_buffer_base(glow::UNIFORM_BUFFER, index, None);
            }
            return self.driver_error(OP);
        };
        let (name, desc) = {
            let entry = self.buffer(OP, buffer)?;
            if !entry.1.usage.contains(GlBufferUsage::UNIFORM) {
                return Err(Self::validation(OP, "buffer lacks uniform usage"));
            }
            entry
        };
        validate_uniform_range(offset, size, desc.size)
            .map_err(|error| Self::validation(OP, error.message()))?;
        let resolved = if size == 0 { desc.size - offset } else { size };
        // SAFETY: current-context contract; usage, alignment, and range were
        // validated against the live allocation.
        unsafe {
            self.bind_buffer_range_wide(
                glow::UNIFORM_BUFFER,
                index,
                Some(name),
                offset,
                resolved,
                OP,
            )?;
        }
        self.driver_error(OP)
    }
}

impl NativeGlProvider {
    /// The discovered limits one binding validation needs.
    pub(super) fn binding_limits(&self) -> GlBindingLimits {
        let limits = self.discovery.limits();
        GlBindingLimits {
            max_texture_units: limits.max_combined_texture_image_units,
            max_uniform_buffer_bindings: limits.max_uniform_buffer_bindings,
            uniform_buffer_offset_alignment: limits.uniform_buffer_offset_alignment,
        }
    }
}

const fn texture_target(target: GlTextureTarget) -> u32 {
    match target {
        GlTextureTarget::D1 => glow::TEXTURE_1D,
        GlTextureTarget::D2 => glow::TEXTURE_2D,
        GlTextureTarget::D2Multisample => glow::TEXTURE_2D_MULTISAMPLE,
        GlTextureTarget::Cube => glow::TEXTURE_CUBE_MAP,
        GlTextureTarget::CubeArray => glow::TEXTURE_CUBE_MAP_ARRAY,
        GlTextureTarget::D3 => glow::TEXTURE_3D,
        GlTextureTarget::D2Array => glow::TEXTURE_2D_ARRAY,
        GlTextureTarget::D2MultisampleArray => glow::TEXTURE_2D_MULTISAMPLE_ARRAY,
    }
}
