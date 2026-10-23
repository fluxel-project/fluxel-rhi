//! Complete fallback sampled objects for unoccupied partially-bound slots.
//!
//! Active WebGL sampler uniforms require complete compatible objects even
//! when control flow never selects the hole. These objects are context-local,
//! cached by native target and sample class, and released with the owner.
//! The public contract forbids shaders from accessing an unbound element.
use super::owner::{BrowserEncoder, BrowserRasterProgram};
use fluxel_rhi_core::api::{Error, Result, TextureSampleType, TextureViewDimension};
use web_sys::WebGl2RenderingContext as Gl;

impl BrowserEncoder {
    pub(crate) fn bind_dummy_sampled_texture(
        &mut self,
        unit: u32,
        dimension: TextureViewDimension,
        sample: TextureSampleType,
        multisampled: bool,
    ) -> Result<()> {
        self.require_raster()?;
        if multisampled {
            return Err(Error::Unsupported("WebGL2 multisampled sampled texture"));
        }
        let target = match dimension {
            TextureViewDimension::D2 => Gl::TEXTURE_2D,
            TextureViewDimension::D2Array => Gl::TEXTURE_2D_ARRAY,
            TextureViewDimension::D3 => Gl::TEXTURE_3D,
            TextureViewDimension::Cube => Gl::TEXTURE_CUBE_MAP,
            _ => return Err(Error::Unsupported("WebGL2 dummy texture dimension")),
        };
        let internal = match sample {
            TextureSampleType::Float | TextureSampleType::UnfilterableFloat => Gl::RGBA8,
            TextureSampleType::Sint => Gl::RGBA8I,
            TextureSampleType::Uint => Gl::RGBA8UI,
            TextureSampleType::Depth => Gl::DEPTH_COMPONENT16,
        };
        let active = Gl::TEXTURE0.checked_add(unit).ok_or(Error::InvalidRange)?;
        let key = (target, internal);
        let existing = self.owner.state.borrow().dummy_textures.get(&key).cloned();
        let raw = if let Some(raw) = existing {
            raw
        } else {
            let mut allocated = None;
            let result = self.with_context("allocate dummy texture", |gl| {
                let Some(texture) = gl.create_texture() else {
                    return;
                };
                gl.active_texture(active);
                gl.bind_texture(target, Some(&texture));
                gl.tex_parameteri(target, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32);
                gl.tex_parameteri(target, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
                match target {
                    Gl::TEXTURE_2D_ARRAY | Gl::TEXTURE_3D => {
                        gl.tex_storage_3d(target, 1, internal, 1, 1, 1)
                    }
                    _ => gl.tex_storage_2d(target, 1, internal, 1, 1),
                }
                allocated = Some(texture);
            });
            if let Err(error) = result {
                if let Some(texture) = allocated {
                    let _ = self.with_context("delete failed dummy texture", |gl| {
                        gl.delete_texture(Some(&texture))
                    });
                }
                return Err(error);
            }
            let texture = allocated.ok_or_else(|| {
                if self.owner.state.borrow().lost {
                    Error::DeviceLost
                } else {
                    Error::OutOfMemory
                }
            })?;
            self.owner
                .state
                .borrow_mut()
                .dummy_textures
                .insert(key, texture.clone());
            texture
        };
        self.with_context("bind dummy texture", |gl| {
            gl.active_texture(active);
            gl.bind_texture(target, Some(&raw));
        })
    }

    pub(crate) fn bind_dummy_sampler(&mut self, unit: u32, comparison: bool) -> Result<()> {
        self.require_raster()?;
        let existing = self
            .owner
            .state
            .borrow()
            .dummy_samplers
            .get(&comparison)
            .cloned();
        let raw = if let Some(raw) = existing {
            raw
        } else {
            let mut allocated = None;
            let result = self.with_context("allocate dummy sampler", |gl| {
                let Some(sampler) = gl.create_sampler() else {
                    return;
                };
                gl.sampler_parameteri(&sampler, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32);
                gl.sampler_parameteri(&sampler, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
                if comparison {
                    gl.sampler_parameteri(
                        &sampler,
                        Gl::TEXTURE_COMPARE_MODE,
                        Gl::COMPARE_REF_TO_TEXTURE as i32,
                    );
                    gl.sampler_parameteri(&sampler, Gl::TEXTURE_COMPARE_FUNC, Gl::LEQUAL as i32);
                }
                allocated = Some(sampler);
            });
            if let Err(error) = result {
                if let Some(sampler) = allocated {
                    let _ = self.with_context("delete failed dummy sampler", |gl| {
                        gl.delete_sampler(Some(&sampler))
                    });
                }
                return Err(error);
            }
            let sampler = allocated.ok_or_else(|| {
                if self.owner.state.borrow().lost {
                    Error::DeviceLost
                } else {
                    Error::OutOfMemory
                }
            })?;
            self.owner
                .state
                .borrow_mut()
                .dummy_samplers
                .insert(comparison, sampler.clone());
            sampler
        };
        self.with_context("bind dummy sampler", |gl| gl.bind_sampler(unit, Some(&raw)))
    }
    /// Uses the linked block's actual byte size, rather than a guessed minimum.
    pub(crate) fn bind_dummy_uniform_buffer(
        &mut self,
        program: &BrowserRasterProgram,
        point: u32,
    ) -> Result<()> {
        self.same_owner(&program.owner)?;
        self.require_raster()?;
        let mut found = None;
        let mut failed = false;
        self.with_context("query dummy uniform block size", |gl| {
            let Some(count) = gl
                .get_program_parameter(&program.raw, Gl::ACTIVE_UNIFORM_BLOCKS)
                .as_f64()
            else {
                failed = true;
                return;
            };
            for index in 0..count as u32 {
                let binding = gl.get_active_uniform_block_parameter(
                    &program.raw,
                    index,
                    Gl::UNIFORM_BLOCK_BINDING,
                );
                let Ok(binding) = binding else {
                    failed = true;
                    return;
                };
                if binding.as_f64() != Some(f64::from(point)) {
                    continue;
                }
                let size = gl.get_active_uniform_block_parameter(
                    &program.raw,
                    index,
                    Gl::UNIFORM_BLOCK_DATA_SIZE,
                );
                let Ok(size) = size else {
                    failed = true;
                    return;
                };
                found = size.as_f64().and_then(|v| {
                    if v > 0. && v <= i32::MAX as f64 && v.fract() == 0. {
                        Some(v as u32)
                    } else {
                        None
                    }
                });
                return;
            }
        })?;
        if failed {
            return Err(Error::Backend("WebGL2 uniform block query failed".into()));
        }
        let size = found.ok_or(Error::InvalidState)?;
        let existing = self
            .owner
            .state
            .borrow()
            .dummy_uniform_buffers
            .get(&size)
            .cloned();
        let buffer = if let Some(buffer) = existing {
            buffer
        } else {
            let mut allocated = None;
            let result = self.with_context("allocate dummy uniform buffer", |gl| {
                let Some(buffer) = gl.create_buffer() else {
                    return;
                };
                gl.bind_buffer(Gl::UNIFORM_BUFFER, Some(&buffer));
                gl.buffer_data_with_i32(Gl::UNIFORM_BUFFER, size as i32, Gl::STATIC_DRAW);
                allocated = Some(buffer);
            });
            if let Err(error) = result {
                if let Some(buffer) = allocated {
                    let _ = self.with_context("delete failed dummy uniform buffer", |gl| {
                        gl.delete_buffer(Some(&buffer))
                    });
                }
                return Err(error);
            }
            let buffer = allocated.ok_or_else(|| {
                if self.owner.state.borrow().lost {
                    Error::DeviceLost
                } else {
                    Error::OutOfMemory
                }
            })?;
            self.owner
                .state
                .borrow_mut()
                .dummy_uniform_buffers
                .insert(size, buffer.clone());
            buffer
        };
        self.with_context("bind dummy uniform buffer", |gl| {
            gl.bind_buffer_base(Gl::UNIFORM_BUFFER, point, Some(&buffer))
        })
    }
}
