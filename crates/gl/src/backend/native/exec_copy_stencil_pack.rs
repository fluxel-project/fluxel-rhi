//! GPU-only conversion from tightly packed stencil bytes to GL packed texels.
use super::super::provider::NativeGlProvider;
use crate::backend::api::{GlCapability, GlError, GlFamilyApi as _, GlFamilyProfile};
use glow::HasContext as _;

/// Three context-owned compute programs for the fixed packed-stencil layouts.
///
/// The pixel count is a uniform, so a transfer's extent never causes shader
/// compilation or linking.  The context profile is immutable for this cache;
/// loss drops the names without issuing deletes, while normal device teardown
/// deletes them through the current context.
pub(crate) struct StencilPackPrograms {
    plain: StencilPackProgram,
    float_depth: StencilPackProgram,
    depth_input: StencilPackProgram,
}

struct StencilPackProgram {
    raw: glow::NativeProgram,
    count: glow::NativeUniformLocation,
}

impl StencilPackPrograms {
    unsafe fn create(gl: &glow::Context, profile: GlFamilyProfile) -> Result<Self, GlError> {
        const OP: &str = "create stencil PBO packing programs";
        let version = if matches!(profile, GlFamilyProfile::Embedded { .. }) {
            "#version 310 es\nprecision highp int;\n"
        } else {
            "#version 430 core\n"
        };
        // SAFETY: the caller owns the current context. Each error branch below
        // deletes every already-built private program before returning.
        unsafe {
            let plain = StencilPackProgram::compile(
                gl,
                version,
                "uint v=(src[i/4u] >> ((i%4u)*8u)) & 255u; dst[i]=v;",
                OP,
            )?;
            let float_depth = match StencilPackProgram::compile(
                gl,
                version,
                "uint v=(src[i/4u] >> ((i%4u)*8u)) & 255u; dst[i*2u]=0u; dst[i*2u+1u]=v;",
                OP,
            ) {
                Ok(program) => program,
                Err(error) => {
                    gl.delete_program(plain.raw);
                    return Err(error);
                }
            };
            let depth_input = match StencilPackProgram::compile(
                gl,
                version,
                "uint v=src[i]; dst[i*2u]=v; dst[i*2u+1u]=0u;",
                OP,
            ) {
                Ok(program) => program,
                Err(error) => {
                    gl.delete_program(plain.raw);
                    gl.delete_program(float_depth.raw);
                    return Err(error);
                }
            };
            Ok(Self {
                plain,
                float_depth,
                depth_input,
            })
        }
    }

    fn select(&self, float_depth: bool, depth_input: bool) -> &StencilPackProgram {
        if depth_input {
            &self.depth_input
        } else if float_depth {
            &self.float_depth
        } else {
            &self.plain
        }
    }

    pub(crate) unsafe fn destroy(self, gl: &glow::Context) {
        // SAFETY: normal teardown has made the exact owner context current.
        unsafe {
            gl.delete_program(self.plain.raw);
            gl.delete_program(self.float_depth.raw);
            gl.delete_program(self.depth_input.raw);
        }
    }
}

impl StencilPackProgram {
    unsafe fn compile(
        gl: &glow::Context,
        version: &str,
        body: &str,
        operation: &'static str,
    ) -> Result<Self, GlError> {
        let source = format!(
            "{version}layout(local_size_x=64) in; layout(std430,binding=0) readonly buffer S {{ uint src[]; }}; layout(std430,binding=1) writeonly buffer D {{ uint dst[]; }}; uniform uint u_count; void main() {{ uint i=gl_GlobalInvocationID.x; if(i>=u_count)return; {body} }}"
        );
        // SAFETY: current-context ownership is the caller's contract; every
        // partially created private object is deleted on its error path.
        let shader = unsafe { gl.create_shader(glow::COMPUTE_SHADER) }
            .map_err(|_| GlError::OutOfMemory { operation })?;
        unsafe {
            gl.shader_source(shader, &source);
            gl.compile_shader(shader);
        }
        if !unsafe { gl.get_shader_compile_status(shader) } {
            let message = unsafe { gl.get_shader_info_log(shader) };
            unsafe { gl.delete_shader(shader) };
            return Err(GlError::Driver { operation, message });
        }
        let raw = match unsafe { gl.create_program() } {
            Ok(raw) => raw,
            Err(_) => {
                unsafe { gl.delete_shader(shader) };
                return Err(GlError::OutOfMemory { operation });
            }
        };
        unsafe {
            gl.attach_shader(raw, shader);
            gl.link_program(raw);
            gl.delete_shader(shader);
        }
        if !unsafe { gl.get_program_link_status(raw) } {
            let message = unsafe { gl.get_program_info_log(raw) };
            unsafe { gl.delete_program(raw) };
            return Err(GlError::Driver { operation, message });
        }
        let Some(count) = (unsafe { gl.get_uniform_location(raw, "u_count") }) else {
            unsafe { gl.delete_program(raw) };
            return Err(GlError::Driver {
                operation,
                message: "linked stencil pack program omitted u_count".into(),
            });
        };
        Ok(Self { raw, count })
    }
}

impl NativeGlProvider {
    /// # Safety
    /// The provider's exact context is current during normal device teardown.
    pub(crate) unsafe fn clear_stencil_pack_programs(&mut self) {
        if let Some(programs) = self.stencil_pack_programs.take() {
            // SAFETY: caller holds the owning context current.
            unsafe { programs.destroy(&self.gl) };
        }
    }

    pub(super) fn stencil_pack_preflight(
        &self,
        op: &'static str,
        count: usize,
        output_bytes: usize,
    ) -> Result<(), GlError> {
        if !matches!(self.discovery.context().profile(),
            GlFamilyProfile::Desktop { major, minor } if major > 4 || (major == 4 && minor >= 3))
            && !matches!(self.discovery.context().profile(), GlFamilyProfile::Embedded { major, minor } if major > 3 || (major == 3 && minor >= 1))
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "plane PBO packing requires GLSL 4.30 or ES 3.10",
            });
        }
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::Compute)
            || !self
                .discovery
                .capabilities()
                .supports(GlCapability::StorageBuffer)
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "stencil PBO packing requires compute and storage buffers",
            });
        }
        let count = u32::try_from(count)
            .map_err(|_| Self::validation(op, "stencil pixel count exceeds shader address"))?;
        if count.div_ceil(64) > self.discovery.limits().max_compute_work_group_count[0]
            || output_bytes as u64 > self.discovery.limits().max_storage_block_size
        {
            return Err(GlError::Unsupported {
                operation: op,
                reason: "stencil pack exceeds compute limits",
            });
        }
        Ok(())
    }
    /// Source is padded to four bytes and contains one tightly packed byte per texel.
    pub(super) fn pack_stencil_gpu(
        &mut self,
        op: &'static str,
        input: glow::NativeBuffer,
        output: glow::NativeBuffer,
        count: usize,
        float_depth: bool,
        depth_input: bool,
    ) -> Result<(), GlError> {
        let count = u32::try_from(count).map_err(|_| Self::validation(op, "pixel count"))?;
        if self.stencil_pack_programs.is_none() {
            // SAFETY: this method owns the current context, and the cache is
            // retained in this provider until normal teardown or context loss.
            self.stencil_pack_programs = Some(unsafe {
                StencilPackPrograms::create(&self.gl, self.discovery.context().profile())?
            });
        }
        let (program, count_location) = {
            let variant = self
                .stencil_pack_programs
                .as_ref()
                .expect("stencil pack cache was installed")
                .select(float_depth, depth_input);
            (variant.raw, variant.count.clone())
        };
        // SAFETY: the cached program and SSBOs belong to this current context;
        // prior indexed and generic bindings and program are restored.
        unsafe {
            let generic = self.generic_storage_buffer;
            let old = [0, 1].map(|i| self.storage_bindings.get(&i).copied());
            let previous = self.current_native_program;
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(input));
            self.gl
                .bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 1, Some(output));
            self.gl.use_program(Some(program));
            self.gl.uniform_1_u32(Some(&count_location), count);
            self.gl.dispatch_compute(count.div_ceil(64), 1, 1);
            self.gl
                .memory_barrier(glow::PIXEL_BUFFER_BARRIER_BIT | glow::SHADER_STORAGE_BARRIER_BIT);
            for (i, binding) in old.into_iter().enumerate() {
                if let Some(binding) =
                    binding.filter(|binding| binding.buffer.is_some() && binding.size > 0)
                {
                    self.bind_buffer_range_wide(
                        glow::SHADER_STORAGE_BUFFER,
                        i as u32,
                        binding.buffer,
                        binding.offset,
                        binding.size,
                        op,
                    )?;
                } else {
                    self.gl.bind_buffer_base(
                        glow::SHADER_STORAGE_BUFFER,
                        i as u32,
                        binding.and_then(|binding| binding.buffer),
                    );
                }
            }
            self.gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, generic);
            self.gl.use_program(previous);
            self.defer_driver_error(op)
        }
    }
}
