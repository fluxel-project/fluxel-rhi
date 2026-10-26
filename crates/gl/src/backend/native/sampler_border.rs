//! Native border sampler variants selected by the sampled texture's numeric class.
use super::provider::NativeGlProvider;
use crate::backend::api::{
    GlDiscoverySnapshot, GlError, GlFamilyApi as _, GlFamilyProfile, GlFormat, GlSamplerApi as _,
    GlSamplerDesc, GlVersion, SamplerId,
};
use core::ffi::c_void;
use glow::HasContext as _;

#[derive(Clone, Copy)]
pub(super) struct NativeSamplerSet {
    pub float: glow::NativeSampler,
    pub sint: Option<glow::NativeSampler>,
    pub uint: Option<glow::NativeSampler>,
}
#[derive(Clone, Copy)]
pub(super) enum NumericClass {
    Float,
    Sint,
    Uint,
}
#[derive(Clone, Copy)]
pub(super) struct SamplerBorderFns {
    signed: unsafe extern "system" fn(u32, u32, *const i32),
    unsigned: unsafe extern "system" fn(u32, u32, *const u32),
}
impl SamplerBorderFns {
    pub(super) unsafe fn load(
        discovery: &GlDiscoverySnapshot,
        loader: impl Fn(&str) -> *const c_void,
    ) -> Option<Self> {
        let ext = discovery.extensions();
        let has = |v| ext.raw_reported_names().any(|x| x == v);
        let mut candidates = Vec::new();
        match discovery.context().profile() {
            GlFamilyProfile::Desktop { .. } => {
                candidates.push(("glSamplerParameterIiv", "glSamplerParameterIuiv"))
            }
            GlFamilyProfile::Embedded { major, minor }
                if GlVersion::new(3, 2).is_met_by(GlVersion::new(major, minor)) =>
            {
                candidates.push(("glSamplerParameterIiv", "glSamplerParameterIuiv"));
            }
            _ => {}
        }
        if matches!(
            discovery.context().profile(),
            GlFamilyProfile::Embedded { .. }
        ) {
            if has("GL_OES_texture_border_clamp") {
                candidates.push(("glSamplerParameterIivOES", "glSamplerParameterIuivOES"));
            }
            if has("GL_EXT_texture_border_clamp") {
                candidates.push(("glSamplerParameterIivEXT", "glSamplerParameterIuivEXT"));
            }
        }
        let (signed, unsigned) = candidates
            .into_iter()
            .map(|(s, u)| (loader(s), loader(u)))
            .find(|(s, u)| !s.is_null() && !u.is_null())?;
        // SAFETY: both loaded functions are defined by the selected current context.
        Some(unsafe {
            Self {
                signed: core::mem::transmute(signed),
                unsigned: core::mem::transmute(unsigned),
            }
        })
    }
}
pub(super) fn numeric_class(format: GlFormat) -> NumericClass {
    match format {
        GlFormat::R8Uint
        | GlFormat::Rg8Uint
        | GlFormat::Rgba8Uint
        | GlFormat::R16Uint
        | GlFormat::Rg16Uint
        | GlFormat::Rgba16Uint
        | GlFormat::Rgb10a2Uint
        | GlFormat::R32Uint
        | GlFormat::R64Uint
        | GlFormat::Rg32Uint
        | GlFormat::Rgba32Uint => NumericClass::Uint,
        GlFormat::R8Sint
        | GlFormat::Rg8Sint
        | GlFormat::Rgba8Sint
        | GlFormat::R16Sint
        | GlFormat::Rg16Sint
        | GlFormat::Rgba16Sint
        | GlFormat::R32Sint
        | GlFormat::Rg32Sint
        | GlFormat::Rgba32Sint => NumericClass::Sint,
        _ => NumericClass::Float,
    }
}
impl NativeGlProvider {
    pub(crate) fn create_border_sampler(
        &mut self,
        desc: GlSamplerDesc,
        axes: [bool; 3],
        color: [f32; 4],
    ) -> Result<SamplerId, GlError> {
        const OP: &str = "create-border-sampler";
        let funcs = self.sampler_border.ok_or(GlError::Unsupported {
            operation: OP,
            reason: "border clamp extension or integer entry points unavailable",
        })?;
        let mut ids = Vec::new();
        for _ in 0..3 {
            match self.create_sampler(desc) {
                Ok(id) => ids.push(id),
                Err(error) => {
                    for id in ids {
                        let _ = self.destroy_sampler(id);
                    }
                    return Err(error);
                }
            }
        }
        let float = self.sampler(OP, ids[0])?;
        let sint = self.sampler(OP, ids[1])?;
        let uint = self.sampler(OP, ids[2])?;
        let signed = color.map(|x| x as i32);
        let unsigned = color.map(|x| x as u32);
        // SAFETY: these samplers and entry points belong to the current context.
        unsafe {
            for name in [float, sint, uint] {
                for (axis, token) in axes.into_iter().zip([
                    glow::TEXTURE_WRAP_S,
                    glow::TEXTURE_WRAP_T,
                    glow::TEXTURE_WRAP_R,
                ]) {
                    if axis {
                        self.gl
                            .sampler_parameter_i32(name, token, glow::CLAMP_TO_BORDER as i32);
                    }
                }
            }
            self.gl
                .sampler_parameter_f32_slice(float, glow::TEXTURE_BORDER_COLOR, &color);
            (funcs.signed)(sint.0.get(), glow::TEXTURE_BORDER_COLOR, signed.as_ptr());
            (funcs.unsigned)(uint.0.get(), glow::TEXTURE_BORDER_COLOR, unsigned.as_ptr());
        }
        if let Err(error) = self.driver_error(OP) {
            for id in ids {
                let _ = self.destroy_sampler(id);
            }
            return Err(error);
        }
        self.samplers.remove(&ids[1]);
        self.samplers.remove(&ids[2]);
        self.samplers.insert(
            ids[0],
            NativeSamplerSet {
                float,
                sint: Some(sint),
                uint: Some(uint),
            },
        );
        Ok(ids[0])
    }
    pub(super) fn bind_sampler_variant(
        &self,
        unit: u32,
        sampler: Option<SamplerId>,
        class: NumericClass,
    ) -> Result<(), GlError> {
        let raw = match sampler {
            None => None,
            Some(id) => {
                self.sampler("bind-sampler", id)?;
                let set = self.samplers.get(&id).expect("validated sampler");
                Some(match class {
                    NumericClass::Float => set.float,
                    NumericClass::Sint => set.sint.unwrap_or(set.float),
                    NumericClass::Uint => set.uint.unwrap_or(set.float),
                })
            }
        };
        // SAFETY: the current context owns the validated sampler.
        unsafe {
            self.gl.bind_sampler(unit, raw);
        }
        self.driver_error("bind-sampler")
    }
}
