//! Immediate polygon, clipping and conservative-raster state.
//!
//! Version/extension checks authenticate native operations. The canonical
//! raster block includes these modes, including restoration to defaults.
use super::provider::NativeGlProvider;
use crate::backend::gl::api::{
    GlDiscoverySnapshot, GlError, GlFamilyProfile, GlPolygonMode, GlPrimitiveTopology,
    GlRasterizationState,
};
use glow::HasContext as _;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ConservativeFlavor {
    #[default]
    None,
    Nv,
    Intel,
}

impl ConservativeFlavor {
    fn select(nv: bool, intel: bool) -> Self {
        if nv {
            Self::Nv
        } else if intel {
            Self::Intel
        } else {
            Self::None
        }
    }

    fn supports(self, state: GlRasterizationState, topology: GlPrimitiveTopology) -> bool {
        !state.conservative
            || match self {
                Self::None => false,
                Self::Nv => true,
                Self::Intel => {
                    state.polygon_mode == GlPolygonMode::Fill
                        && matches!(
                            topology,
                            GlPrimitiveTopology::Triangles | GlPrimitiveTopology::TriangleStrip
                        )
                }
            }
    }

    fn token(self) -> Option<u32> {
        match self {
            Self::None => None,
            Self::Nv => Some(0x9346),
            Self::Intel => Some(0x83fe),
        }
    }
}

#[derive(Default)]
pub(super) struct NativeRasterization {
    polygon_mode: Option<unsafe extern "system" fn(u32, u32)>,
    point_mode: bool,
    depth_clamp: bool,
    conservative: ConservativeFlavor,
    output_interface: Option<unsafe extern "system" fn(u32, u32, u32, *mut i32)>,
    output_resource: Option<super::context::GetProgramResourceFn>,
    frag_data_index: Option<unsafe extern "system" fn(u32, *const i8) -> i32>,
}

impl NativeRasterization {
    /// # Safety
    /// The loader resolves procedures belonging to the current context.
    pub(super) unsafe fn load(
        discovery: &GlDiscoverySnapshot,
        loader: &impl Fn(&str) -> *const core::ffi::c_void,
    ) -> Self {
        let desktop = matches!(
            discovery.context().profile(),
            GlFamilyProfile::Desktop { .. }
        );
        let has = |name| {
            discovery
                .extensions()
                .raw_reported_names()
                .any(|value| value == name)
        };
        let symbol = if desktop {
            Some("glPolygonMode")
        } else if has("GL_ANGLE_polygon_mode") {
            Some("glPolygonModeANGLE")
        } else {
            None
        };
        let polygon_mode = symbol.map(loader).filter(|ptr| !ptr.is_null()).map(|ptr| {
            // SAFETY: core and ANGLE procedures share the GLenum pair ABI.
            unsafe { core::mem::transmute(ptr) }
        });
        let interfaces = matches!(discovery.context().profile(), GlFamilyProfile::Desktop { major, minor } if (major, minor) >= (4, 3))
            || matches!(discovery.context().profile(), GlFamilyProfile::Embedded { major, minor } if (major, minor) >= (3, 1))
            || (desktop && has("GL_ARB_program_interface_query"));
        let interface = if interfaces {
            loader("glGetProgramInterfaceiv")
        } else {
            std::ptr::null()
        };
        let resource = if interfaces {
            loader("glGetProgramResourceiv")
        } else {
            std::ptr::null()
        };
        // SAFETY: these are the named core GL procedures with the signatures declared above.
        let output_interface =
            (!interface.is_null()).then(|| unsafe { core::mem::transmute(interface) });
        // SAFETY: the pointer names glGetProgramResourceiv in this current context.
        let output_resource =
            (!resource.is_null()).then(|| unsafe { core::mem::transmute(resource) });
        let index = if desktop {
            loader("glGetFragDataIndex")
        } else if has("GL_EXT_blend_func_extended") {
            loader("glGetFragDataIndexEXT")
        } else {
            std::ptr::null()
        };
        // SAFETY: the core and EXT procedures have identical signatures.
        let frag_data_index = (!index.is_null()).then(|| unsafe { core::mem::transmute(index) });
        Self {
            polygon_mode,
            point_mode: desktop,
            depth_clamp: desktop || has("GL_EXT_depth_clamp"),
            conservative: ConservativeFlavor::select(
                has("GL_NV_conservative_raster"),
                has("GL_INTEL_conservative_rasterization"),
            ),
            output_interface,
            output_resource,
            frag_data_index,
        }
    }

    pub(super) fn supports(&self, state: GlRasterizationState) -> bool {
        (state.polygon_mode == GlPolygonMode::Fill || self.polygon_mode.is_some())
            && (state.polygon_mode != GlPolygonMode::Point || self.point_mode)
            && (!state.depth_clamp || self.depth_clamp)
            && (!state.conservative || self.conservative != ConservativeFlavor::None)
            && (!state.conservative
                || self.conservative != ConservativeFlavor::Intel
                || state.polygon_mode == GlPolygonMode::Fill)
    }
}

impl NativeGlProvider {
    pub(crate) fn has_secondary_output(
        &self,
        program: crate::backend::gl::api::ProgramId,
    ) -> Result<bool, GlError> {
        let entry = self.program("reflect dual-source output", program)?;
        let (Some(interface), Some(resource)) = (
            self.rasterization.output_interface,
            self.rasterization.output_resource,
        ) else {
            let index = self
                .rasterization
                .frag_data_index
                .ok_or(GlError::Unsupported {
                    operation: "reflect dual-source output",
                    reason: "fragment output index query unavailable",
                })?;
            let crate::backend::gl::api::GlProgramKind::Raster { fragment, .. } =
                &entry.descriptor.kind
            else {
                return Ok(false);
            };
            // Candidate spelling is only a discovery aid: linked GL reflection is
            // authoritative, so inactive code and macro branches cannot create
            // a false positive. Token pasting can hide the final spelling and is
            // rejected if no reflected candidate is found.
            let mut names = std::collections::BTreeSet::new();
            for token in fragment
                .text
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            {
                if token
                    .as_bytes()
                    .first()
                    .is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_')
                {
                    names.insert(token);
                }
            }
            for name in names {
                let c_name = std::ffi::CString::new(name).expect("ASCII identifier");
                // SAFETY: program belongs to the current context and c_name is NUL terminated.
                let secondary = unsafe { index(entry.raw.0.get(), c_name.as_ptr()) } == 1;
                // SAFETY: same live linked program and identifier.
                if secondary && unsafe { self.gl.get_frag_data_location(entry.raw, name) } == 0 {
                    return Ok(true);
                }
            }
            if fragment.text.contains("##") {
                // GL's linked query remains authoritative. Try combinations of
                // source identifiers to cover ordinary two-token paste even if
                // preprocessor conditions depend on driver-defined macros.
                let names: std::collections::BTreeSet<_> = fragment
                    .text
                    .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .filter(|s| {
                        s.as_bytes()
                            .first()
                            .is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_')
                    })
                    .collect();
                if names.len() <= 128 {
                    for left in &names {
                        for right in &names {
                            let candidate = format!("{left}{right}");
                            let c_name = std::ffi::CString::new(candidate.as_str())
                                .expect("ASCII identifier");
                            // SAFETY: program belongs to this context and c_name is NUL terminated.
                            if unsafe { index(entry.raw.0.get(), c_name.as_ptr()) } == 1
                                && unsafe { self.gl.get_frag_data_location(entry.raw, &candidate) }
                                    == 0
                            {
                                return Ok(true);
                            }
                        }
                    }
                }
                return Err(GlError::Unsupported {
                    operation: "reflect dual-source output",
                    reason: "output name generated by complex token pasting cannot be enumerated on this profile",
                });
            }
            return Ok(false);
        };
        let program = entry.raw.0.get();
        let mut count = 0;
        // SAFETY: the current context owns the linked program and both calls write valid output slots.
        unsafe {
            interface(
                program,
                glow::PROGRAM_OUTPUT,
                glow::ACTIVE_RESOURCES,
                &mut count,
            );
        }
        for index in 0..count.max(0) as u32 {
            let props = [glow::LOCATION, glow::LOCATION_INDEX];
            let mut values = [-1, -1];
            let mut written = 0;
            // SAFETY: the resource index is within the queried count; arrays have two elements.
            unsafe {
                resource(
                    program,
                    glow::PROGRAM_OUTPUT,
                    index,
                    2,
                    props.as_ptr(),
                    2,
                    &mut written,
                    values.as_mut_ptr(),
                );
            }
            if written == 2 && values == [0, 1] {
                return Ok(true);
            }
        }
        Ok(false)
    }
    pub(crate) fn dual_source_limit(&self) -> u32 {
        let profile = self.discovery.context().profile();
        let available = matches!(profile, GlFamilyProfile::Desktop { major, minor } if (major, minor) >= (3, 3))
            || self
                .discovery
                .extensions()
                .raw_reported_names()
                .any(|name| {
                    name == "GL_ARB_blend_func_extended" || name == "GL_EXT_blend_func_extended"
                });
        if !available {
            return 0;
        }
        // SAFETY: the provider's context is current and the enum is legal after the capability check.
        unsafe { self.gl.get_parameter_i32(0x88FC).max(0) as u32 }
    }
    /// # Safety
    /// The provider context is current. Optional polygon-offset capabilities
    /// were loaded from this context and retain their matching enum legality.
    pub(super) unsafe fn set_polygon_offset_enabled(&self, enabled: bool) {
        // SAFETY: forwarded current-context contract; only supported targets
        // are changed, including restoring line/point modes on pipeline switch.
        unsafe {
            for (supported, token) in [
                (true, glow::POLYGON_OFFSET_FILL),
                (
                    self.rasterization.polygon_mode.is_some(),
                    glow::POLYGON_OFFSET_LINE,
                ),
                (self.rasterization.point_mode, glow::POLYGON_OFFSET_POINT),
            ] {
                if supported {
                    if enabled {
                        self.gl.enable(token);
                    } else {
                        self.gl.disable(token);
                    }
                }
            }
        }
    }

    pub(crate) fn supports_rasterization(&self, state: GlRasterizationState) -> bool {
        self.rasterization.supports(state)
    }

    pub(crate) fn supports_rasterization_topology(
        &self,
        state: GlRasterizationState,
        topology: GlPrimitiveTopology,
    ) -> bool {
        self.rasterization.supports(state)
            && self.rasterization.conservative.supports(state, topology)
    }

    /// # Safety
    /// The provider context is current during installation of fixed state.
    pub(super) unsafe fn apply_rasterization(
        &self,
        state: GlRasterizationState,
    ) -> Result<(), GlError> {
        if !self.supports_rasterization(state) {
            return Err(GlError::Unsupported {
                operation: "set-rasterization",
                reason: "native polygon, depth-clamp or conservative-raster capability is unavailable",
            });
        }
        // SAFETY: the context is current; optional enums/procedures are issued
        // only when the matching core version or extension was discovered.
        unsafe {
            if let Some(polygon_mode) = self.rasterization.polygon_mode {
                polygon_mode(
                    glow::FRONT_AND_BACK,
                    match state.polygon_mode {
                        GlPolygonMode::Fill => glow::FILL,
                        GlPolygonMode::Line => glow::LINE,
                        GlPolygonMode::Point => glow::POINT,
                    },
                );
            }
            for (supported, enabled, token) in
                [(self.rasterization.depth_clamp, state.depth_clamp, 0x864F)]
            {
                if supported {
                    if enabled {
                        self.gl.enable(token);
                    } else {
                        self.gl.disable(token);
                    }
                }
            }
            if let Some(token) = self.rasterization.conservative.token() {
                if state.conservative {
                    self.gl.enable(token);
                } else {
                    self.gl.disable(token);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod conservative_tests {
    use super::*;

    #[test]
    fn extension_selection_and_topology_constraints() {
        let mut state = GlRasterizationState {
            polygon_mode: GlPolygonMode::Fill,
            depth_clamp: false,
            conservative: true,
        };
        assert_eq!(
            ConservativeFlavor::select(true, true),
            ConservativeFlavor::Nv
        );
        assert_eq!(
            ConservativeFlavor::select(false, true).token(),
            Some(0x83fe)
        );
        assert!(!ConservativeFlavor::None.supports(state, GlPrimitiveTopology::Triangles));
        assert!(ConservativeFlavor::Intel.supports(state, GlPrimitiveTopology::Triangles));
        assert!(ConservativeFlavor::Intel.supports(state, GlPrimitiveTopology::TriangleStrip));
        assert!(!ConservativeFlavor::Intel.supports(state, GlPrimitiveTopology::Lines));
        assert!(ConservativeFlavor::Nv.supports(state, GlPrimitiveTopology::Lines));
        state.polygon_mode = GlPolygonMode::Line;
        assert!(!ConservativeFlavor::Intel.supports(state, GlPrimitiveTopology::Triangles));
        state.conservative = false;
        assert!(ConservativeFlavor::Intel.supports(state, GlPrimitiveTopology::Lines));
    }
}
