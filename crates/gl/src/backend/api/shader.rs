//! Platform-neutral shader, program, and pipeline-layout vocabulary.

use std::collections::BTreeSet;

use super::{
    GlError, GlFamilyApi, GlFamilyProfile, GlStorageBufferUsage, GlStorageImageAccess, ProgramId,
    ShaderId,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlShaderStage {
    Vertex,
    Fragment,
    Compute,
}

/// Content identity for a source accepted by this backend instance.
///
/// It is backend-private because Core describes source bytes rather than a
/// cross-backend shader artifact ABI. It identifies immutable source only; it
/// is never a GL object name and does not participate in resource provenance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlShaderKey(pub(crate) [u8; 32]);

/// The exact GLSL family and version emitted by the GL lowering pass.
///
/// This is not an authoring-language choice. A `GlShaderSource` has already
/// been lowered from the private validated artifact and is ready for one GL
/// profile only.
/// Exact GLSL target selected by GL lowering.  This is backend-private because
/// the public artifact carries code form while a context selects a concrete
/// desktop/ES version from discovery evidence.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlShaderDialect {
    Desktop { version: u16 },
    Embedded { version: u16 },
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlShaderSource {
    pub stage: GlShaderStage,
    pub dialect: GlShaderDialect,
    pub entry_point: String,
    pub source_hash: GlShaderKey,
    pub text: String,
    /// Desktop GL SPIR-V artifact. Text remains empty for this route.
    pub spirv: Option<GlSpirvSource>,
    pub debug_name: Option<String>,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlSpirvSource {
    pub words: Vec<u32>,
    pub specialization: Vec<(u32, u32)>,
}
/// Fluxel's logical binding key. It deliberately is not a GL binding point.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GlBindingLocation {
    pub group: u32,
    pub binding: u32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlShaderResourceKind {
    UniformBuffer,
    Sampler,
    Texture,
    CombinedTextureSampler,
    /// A shader storage buffer, read or written by a compute stage.
    ///
    /// The two storage kinds are the only ones a raster program can never
    /// declare, and that is a fact about a shading language rather than about
    /// this crate: the WebGL2 profile's GLSL ES 300 has no storage qualifier at
    /// all.  Nothing here judges which profile admits them, for the same reason
    /// nothing here judges compute capability -- that is the provider's call,
    /// made against the linked program, and the two providers answer it
    /// differently.
    ///
    /// # Why each storage kind carries its access
    ///
    /// The payload is the access the *declaration* states, and it is here
    /// because a layout is what a caller reads to build a native bind: a storage
    /// buffer bind has a usage and a storage image bind has an access, so a kind
    /// that named neither would describe a binding nothing could bind.  It is a
    /// fact about the shader rather than about the resource -- the same buffer
    /// can be bound read-only to one program and read-write to another -- which
    /// is why it belongs on the declaration and not on the object.
    ///
    /// Neither provider reads it, and that is not an oversight: the native
    /// reflection resolves a block by name off the linked program and the
    /// browser provider refuses both kinds outright, so each has a better source
    /// than the descriptor for what it needs.  The consumer is the compatibility
    /// adapter, which has only the descriptor.
    StorageBuffer(GlStorageBufferUsage),
    /// A shader storage image, read or written by a compute stage.
    StorageImage(GlStorageImageAccess),
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlSampledDimension {
    D1,
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlSampledKind {
    Float,
    Sint,
    Uint,
    Depth,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlSampledBindingType {
    Texture {
        dimension: GlSampledDimension,
        kind: GlSampledKind,
        multisampled: bool,
    },
    Sampler {
        comparison: bool,
    },
}
impl GlSampledBindingType {
    pub(crate) fn matches_uniform(self, ty: u32) -> bool {
        use GlSampledDimension as D;
        use GlSampledKind as K;
        let (dimension, kind, multisampled, shadow) = match ty {
            0x8B5D => (D::D1, K::Float, false, false),
            0x8B5E => (D::D2, K::Float, false, false),
            0x8B5F => (D::D3, K::Float, false, false),
            0x8B60 => (D::Cube, K::Float, false, false),
            0x8B62 => (D::D2, K::Float, false, true),
            0x8B61 => (D::D1, K::Float, false, true),
            0x8DC1 => (D::D2Array, K::Float, false, false),
            0x8DC4 => (D::D2Array, K::Float, false, true),
            0x8DC5 => (D::Cube, K::Float, false, true),
            0x900C => (D::CubeArray, K::Float, false, false),
            0x900D => (D::CubeArray, K::Float, false, true),
            0x900E => (D::CubeArray, K::Sint, false, false),
            0x900F => (D::CubeArray, K::Uint, false, false),
            0x8DCA => (D::D2, K::Sint, false, false),
            0x8DC9 => (D::D1, K::Sint, false, false),
            0x8DCB => (D::D3, K::Sint, false, false),
            0x8DCC => (D::Cube, K::Sint, false, false),
            0x8DCF => (D::D2Array, K::Sint, false, false),
            0x8DD2 => (D::D2, K::Uint, false, false),
            0x8DD1 => (D::D1, K::Uint, false, false),
            0x8DD3 => (D::D3, K::Uint, false, false),
            0x8DD4 => (D::Cube, K::Uint, false, false),
            0x8DD7 => (D::D2Array, K::Uint, false, false),
            0x9108 => (D::D2, K::Float, true, false),
            0x9109 => (D::D2, K::Sint, true, false),
            0x910A => (D::D2, K::Uint, true, false),
            0x910B => (D::D2Array, K::Float, true, false),
            0x910C => (D::D2Array, K::Sint, true, false),
            0x910D => (D::D2Array, K::Uint, true, false),
            _ => return false,
        };
        match self {
            Self::Texture {
                dimension: d,
                kind: k,
                multisampled: m,
            } => {
                d == dimension
                    && m == multisampled
                    && (!shadow || k == K::Depth)
                    && (k == kind || (k == K::Depth && kind == K::Float))
            }
            Self::Sampler { comparison } => !multisampled && comparison == shadow,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/backend/api/shader__sampled_type_tests.rs"]
mod sampled_type_tests;
/// A logical RHI resource declaration, independent of program-link results.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlLogicalBinding {
    /// All stage-local GLSL declarations generated for this logical binding.
    /// A shared RHI UBO can legitimately have different generated names in
    /// the vertex and fragment sources.
    pub names: Vec<String>,
    pub location: GlBindingLocation,
    /// Fixed binding-array element represented by this linked declaration.
    pub element: Option<u32>,
    pub kind: GlShaderResourceKind,
    pub array_count: u32,
    /// Whether the Core declaration is a fixed array, including length one.
    pub fixed_array: bool,
    /// The other logical half of a WGSL texture/sampler pair. Both entries
    /// resolve to one GLSL sampler uniform and therefore one texture unit.
    pub pair: Option<GlBindingLocation>,
    /// Explicit GLSL image-unit binding emitted by Naga for storage images.
    /// ES image uniforms cannot be assigned with `glUniform1i`.
    pub image_unit: Option<u32>,
    pub sampled_type: Option<GlSampledBindingType>,
    /// Storage image dimension and numeric class used by linked type validation.
    pub image_type: Option<(GlSampledDimension, GlSampledKind)>,
}
/// An executable GL assignment selected after program link.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlExecutableBindingLocation {
    UniformBlock(u32),
    TextureUnit(u32),
    /// The linked program's own index for a shader storage block.
    ///
    /// An index rather than a binding point, for [`Self::UniformBlock`]'s
    /// reason: what a logical binding names and where the driver binds it are
    /// two facts, and this is the first of them.
    StorageBlock(u32),
    /// An image unit, assigned by the provider that reflected it.
    ///
    /// [`Self::TextureUnit`]'s counterpart for the storage domain.  The two are
    /// separate variants rather than one unit type because they are separate
    /// namespaces: an image unit is not a texture unit, and numbering starts
    /// again for it.
    ImageUnit(u32),
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlExecutableBindingAssignment {
    pub logical: GlBindingLocation,
    /// The corresponding fixed binding-array element, if any.
    pub element: Option<u32>,
    pub executable: GlExecutableBindingLocation,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlVertexInputReflection {
    pub name: String,
    pub location: u32,
    pub columns: u8,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlFragmentOutputReflection {
    pub name: String,
    pub location: u32,
}
/// Immutable link evidence. Logical bindings and executable locations remain
/// distinct so a cache cannot mistake a GL assignment for a RHI contract.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlProgramReflection {
    pub vertex_inputs: Vec<GlVertexInputReflection>,
    pub fragment_outputs: Vec<GlFragmentOutputReflection>,
    pub assignments: Vec<GlExecutableBindingAssignment>,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlPipelineLayout {
    pub bindings: Vec<GlLogicalBinding>,
}
/// Explicit program routing. A linked program is raster or compute, never both.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlProgramKind {
    Raster {
        vertex: GlShaderSource,
        fragment: GlShaderSource,
    },
    Compute {
        shader: GlShaderSource,
    },
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlProgramDescriptor {
    pub kind: GlProgramKind,
    pub layout: GlPipelineLayout,
    pub debug_name: Option<String>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlShaderValidationError {
    MixedShaderFormats,
    WrongStage,
    EmptySource,
    EmptyEntryPoint,
    DialectProfileMismatch,
    ZeroArrayCount,
    DuplicateLogicalBinding,
    DuplicateLogicalAssignment,
    DuplicateExecutableAssignment,
    MissingStorageImageUnit,
    StorageImageUnitMismatch,
    DuplicateVertexLocation,
    DuplicateFragmentOutputLocation,
    ZeroVertexColumns,
}

impl GlPipelineLayout {
    pub(crate) fn validate(&self) -> Result<(), GlShaderValidationError> {
        let mut locations = BTreeSet::new();
        for binding in &self.bindings {
            if binding.array_count == 0 {
                return Err(GlShaderValidationError::ZeroArrayCount);
            }
            if !locations.insert((binding.location, binding.element)) {
                return Err(GlShaderValidationError::DuplicateLogicalBinding);
            }
        }
        Ok(())
    }
}
impl GlProgramDescriptor {
    pub(crate) fn validate(&self) -> Result<(), GlShaderValidationError> {
        match &self.kind {
            GlProgramKind::Raster { vertex, fragment } => {
                if vertex.spirv.is_some() != fragment.spirv.is_some() {
                    return Err(GlShaderValidationError::MixedShaderFormats);
                }
                if vertex.stage != GlShaderStage::Vertex
                    || fragment.stage != GlShaderStage::Fragment
                {
                    return Err(GlShaderValidationError::WrongStage);
                }
                vertex.validate()?;
                fragment.validate()?;
            }
            GlProgramKind::Compute { shader } => {
                if shader.stage != GlShaderStage::Compute {
                    return Err(GlShaderValidationError::WrongStage);
                }
                shader.validate()?;
            }
        }
        self.layout.validate()
    }

    /// Validates that every already-lowered source targets this exact context.
    pub(crate) fn validate_for(
        &self,
        profile: GlFamilyProfile,
    ) -> Result<(), GlShaderValidationError> {
        self.validate()?;
        match &self.kind {
            GlProgramKind::Raster { vertex, fragment } => {
                vertex.validate_for(profile)?;
                fragment.validate_for(profile)
            }
            GlProgramKind::Compute { shader } => shader.validate_for(profile),
        }
    }
}
impl GlShaderSource {
    pub(crate) fn validate(&self) -> Result<(), GlShaderValidationError> {
        if self.text.is_empty() && self.spirv.is_none() {
            return Err(GlShaderValidationError::EmptySource);
        }
        if self.entry_point.is_empty() {
            return Err(GlShaderValidationError::EmptyEntryPoint);
        }
        Ok(())
    }

    /// Checks that the lowered dialect belongs to, and is no newer than, this
    /// context profile. Desktop GL keeps compatibility for earlier core GLSL
    /// versions (for example GLSL 400 on a GL 4.6 context); requiring exact
    /// equality would reject a valid source artifact solely because discovery
    /// observed a newer driver than the asset was authored for.
    pub(crate) fn validate_for(
        &self,
        profile: GlFamilyProfile,
    ) -> Result<(), GlShaderValidationError> {
        self.validate()?;
        if self.spirv.is_some() {
            return matches!(profile, GlFamilyProfile::Desktop { .. })
                .then_some(())
                .ok_or(GlShaderValidationError::DialectProfileMismatch);
        }
        match (profile, self.dialect) {
            (GlFamilyProfile::WebGl2, GlShaderDialect::Embedded { version: 300 }) => Ok(()),
            (
                GlFamilyProfile::Embedded { major: 3, minor },
                GlShaderDialect::Embedded { version },
            ) if version <= 300 + u16::from(minor) * 10 => Ok(()),
            (
                GlFamilyProfile::Desktop { major: 4, minor },
                GlShaderDialect::Desktop { version },
            ) if version >= 400 && version <= 400 + u16::from(minor) * 10 => Ok(()),
            _ => Err(GlShaderValidationError::DialectProfileMismatch),
        }
    }
}
impl GlProgramReflection {
    pub(crate) fn validate_against(
        &self,
        layout: &GlPipelineLayout,
    ) -> Result<(), GlShaderValidationError> {
        layout.validate()?;
        let lookup = |location, element| {
            layout.bindings.iter().find(|binding| {
                binding.location == location
                    && (binding.element == element
                        || (binding.element.is_none()
                            && binding.fixed_array
                            && element.is_some_and(|index| index < binding.array_count)))
            })
        };
        let mut executable = std::collections::BTreeMap::new();
        for assignment in &self.assignments {
            if lookup(assignment.logical, assignment.element).is_none() {
                return Err(GlShaderValidationError::DuplicateLogicalAssignment);
            }
            if let Some(previous) = executable.insert(
                assignment.executable,
                (assignment.logical, assignment.element),
            ) {
                let current = lookup(assignment.logical, assignment.element);
                let prior = lookup(previous.0, previous.1);
                let paired = current.zip(prior).is_some_and(|(a, b)| {
                    matches!(
                        assignment.executable,
                        GlExecutableBindingLocation::TextureUnit(_)
                    ) && (a.pair == Some(previous.0)
                        || (matches!(
                            (a.kind, b.kind),
                            (GlShaderResourceKind::Texture, GlShaderResourceKind::Sampler)
                                | (GlShaderResourceKind::Sampler, GlShaderResourceKind::Texture)
                        ) && a.names.iter().any(|name| b.names.contains(name))))
                });
                if previous != (assignment.logical, assignment.element) && !paired {
                    return Err(GlShaderValidationError::DuplicateExecutableAssignment);
                }
            }
        }
        for binding in &layout.bindings {
            if matches!(binding.kind, GlShaderResourceKind::StorageImage(_)) {
                let Some(unit) = binding.image_unit else {
                    continue;
                };
                if !self.assignments.iter().any(|assignment| {
                    assignment.logical == binding.location
                        && assignment.element == binding.element
                        && assignment.executable == GlExecutableBindingLocation::ImageUnit(unit)
                }) {
                    return Err(GlShaderValidationError::StorageImageUnitMismatch);
                }
            }
        }
        let mut inputs = BTreeSet::new();
        for input in &self.vertex_inputs {
            if input.columns == 0 {
                return Err(GlShaderValidationError::ZeroVertexColumns);
            }
            if !inputs.insert(input.location) {
                return Err(GlShaderValidationError::DuplicateVertexLocation);
            }
        }
        let mut outputs = BTreeSet::new();
        if self
            .fragment_outputs
            .iter()
            .any(|output| !outputs.insert(output.location))
        {
            return Err(GlShaderValidationError::DuplicateFragmentOutputLocation);
        }
        Ok(())
    }
}
/// Shader, program, and reflection domain.
///
/// `create_program` routes on the descriptor's explicit [`GlProgramKind`]; a
/// provider accepts the compute kind only after its discovery snapshot proved
/// the compute capability. Reflection for compute programs is empty until the
/// dedicated reflection wave lands.
pub(crate) trait GlShaderApi: GlFamilyApi {
    fn create_shader(&mut self, source: &GlShaderSource) -> Result<ShaderId, GlError>;
    fn destroy_shader(&mut self, shader: ShaderId) -> Result<(), GlError>;
    fn create_program(
        &mut self,
        descriptor: &GlProgramDescriptor,
    ) -> Result<(ProgramId, GlProgramReflection), GlError>;
    fn destroy_program(&mut self, program: ProgramId) -> Result<(), GlError>;
}
#[cfg(test)]
#[path = "../../tests/backend/api/shader__tests.rs"]
mod tests;
