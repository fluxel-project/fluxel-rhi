//! GL backend identity and thread-local native ownership.
//!
//! The state is `Rc`-backed intentionally: a GL context is bound to one owner
//! thread. Resource wrappers retain this state so their native handles cannot
//! outlive the context that created them. Native calls first make the platform
//! context current and then issue directly through its provider; this module
//! never queues or replays a command stream.

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) use super::completion::GlCompletion;
pub(crate) use super::encoder::GlEncoder;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(crate) use super::encoder::{GlOffscreenResolve, GlSurfaceDepthPass};

/// Static backend marker for desktop GL, GLES, and WebGL2.
pub struct Gl;

/// A concrete adopted GL context from which one Core device is opened.
#[derive(Clone)]
pub struct GlInstance(pub(crate) Rc<GlDeviceState>);

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl GlInstance {
    /// Adopts a platform-owned context that is used synchronously on this
    /// thread. The dispatch table is loaded once, but each command still makes
    /// the context current before issuing a `glow` call.
    pub(crate) fn from_current_context(
        mut context: crate::backend::gl::native::NativePlatformContext,
    ) -> fluxel_rhi_core::api::Result<Self> {
        use crate::backend::gl::native::NativeContextOwner as _;

        context
            .make_current("adopt GL context")
            .map_err(|error| fluxel_rhi_core::api::Error::Backend(format!("{error:?}")))?;
        let gl = context
            .load_glow()
            .map_err(|error| fluxel_rhi_core::api::Error::Backend(format!("{error:?}")))?;
        let unmap_buffer = context
            .load_unmap_buffer()
            .map_err(|error| fluxel_rhi_core::api::Error::Backend(format!("{error:?}")))?;
        let stamp = context.context_stamp();
        // SAFETY: `make_current` above established that this exact platform
        // context is current and exclusively used by this thread. The provider
        // is retained beside that owner and every later operation repeats the
        // currentness transition before touching its GL names.
        let provider = unsafe {
            crate::backend::gl::native::provider::NativeGlProvider::from_current(
                gl,
                stamp,
                unmap_buffer,
            )
        }
        .map_err(|error| fluxel_rhi_core::api::Error::Backend(format!("{error:?}")))?;
        Ok(Self(Rc::new(GlDeviceState {
            context: RefCell::new(context),
            provider: RefCell::new(provider),
        })))
    }
}
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
impl GlInstance {
    /// Adopts one browser-owned WebGL2 context. BrowserOwner retains the DOM
    /// context and issues calls directly on its owner thread; no JS handle is
    /// copied into the portable Core API.
    pub(crate) fn from_browser_owner(owner: crate::backend::gl::browser::BrowserOwner) -> Self {
        Self(Rc::new(GlDeviceState { owner }))
    }
}

/// Native GL device retained by every backend resource.
#[derive(Clone)]
pub struct GlDevice(pub(crate) Rc<GlDeviceState>);

/// Thread-local state shared by one current GL context generation.
pub(crate) struct GlDeviceState {
    /// Platform context ownership remains private and is never put in Core.
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) context: RefCell<crate::backend::gl::native::NativePlatformContext>,
    /// The provider owns the loader-resolved dispatch table and the private GL
    /// object tables. It is accessed only through [`Self::with_native`], which
    /// makes the associated platform context current first.
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) provider: RefCell<crate::backend::gl::native::provider::NativeGlProvider>,
    /// Browser ownership is already `Rc<RefCell<_>>` and thread-affine.
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub(crate) owner: crate::backend::gl::browser::BrowserOwner,
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
    pub(crate) _private: (),
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl GlDeviceState {
    /// Executes a platform-context operation without borrowing the GL object
    /// provider. Presentation uses this after an image lease is irrevocably
    /// handed off, so a swap failure cannot be reclassified as rejection.
    pub(crate) fn with_context<T>(
        &self,
        operation: &'static str,
        f: impl FnOnce(
            &mut crate::backend::gl::native::NativePlatformContext,
        ) -> Result<T, crate::backend::gl::api::GlError>,
    ) -> Result<T, crate::backend::gl::api::GlError> {
        let mut context = self.context.try_borrow_mut().map_err(|_| {
            crate::backend::gl::api::GlError::Driver {
                operation,
                message: "GL platform context is already in use".into(),
            }
        })?;
        f(&mut context)
    }

    /// Executes one native operation while this device's context is current.
    ///
    /// The borrow ordering is deliberate: currentness is established before a
    /// provider can resolve or issue any object name. `Rc` and `RefCell` keep
    /// this state thread-affine, so no other Rust thread can concurrently make
    /// the context current or mutate its object tables.
    pub(crate) fn with_native<T>(
        &self,
        operation: &'static str,
        f: impl FnOnce(
            &mut crate::backend::gl::native::provider::NativeGlProvider,
        ) -> Result<T, crate::backend::gl::api::GlError>,
    ) -> Result<T, crate::backend::gl::api::GlError> {
        use crate::backend::gl::native::NativeContextOwner as _;

        let mut context = self.context.try_borrow_mut().map_err(|_| {
            crate::backend::gl::api::GlError::Driver {
                operation,
                message: "GL context is already issuing a command".into(),
            }
        })?;
        context.make_current(operation)?;
        let mut provider = self.provider.try_borrow_mut().map_err(|_| {
            crate::backend::gl::api::GlError::Driver {
                operation,
                message: "GL provider is already issuing a command".into(),
            }
        })?;
        f(&mut provider)
    }

    /// Names the single adapter represented by this adopted GL context.
    ///
    /// GL has no portable physical-device enumeration after context creation.
    /// Discovery's immutable driver strings are therefore the truthful facts
    /// used by the public instance enumeration result.
    pub(crate) fn adapter_name(&self) -> Result<String, crate::backend::gl::api::GlError> {
        use crate::backend::gl::api::GlFamilyApi as _;

        self.with_native("describe GL adapter", |provider| {
            let context = provider.discovery().context();
            Ok(format!(
                "{} {} ({})",
                context.vendor(),
                context.renderer(),
                context.version()
            ))
        })
    }
}

#[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
impl GlDeviceState {
    /// Featureless builds retain the marker types but cannot adopt a context.
    /// A browser-specific implementation replaces this on the wasm WebGL2
    /// build; no synthetic adapter is reported on unsupported targets.
    pub(crate) fn adapter_name(&self) -> Result<String, crate::backend::gl::api::GlError> {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            let context = self.owner.discovery_snapshot();
            let info = context.context();
            return Ok(format!(
                "{} {} ({})",
                info.vendor(),
                info.renderer(),
                info.version()
            ));
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        Err(crate::backend::gl::api::GlError::Unsupported {
            operation: "describe GL adapter",
            reason: "no GL platform context feature is enabled",
        })
    }
}

macro_rules! native_handle {
    ($($name:ident),+ $(,)?) => {$(
        #[derive(Clone)]
        pub struct $name(pub(crate) Rc<GlDeviceState>);
    )+};
}

native_handle!(
    GlMeshPipeline,
    GlRayPipeline,
    GlPipelineCache,
    GlAcceleration,
    GlSparseAllocation,
);

/// GL query names retained with their owning context and Core query kind.
///
/// Query slots are reset logically because GL overwrites a query's value when
/// it begins again; the per-slot flags prevent resolving stale driver values.
#[derive(Clone)]
pub struct GlQuerySet(pub(crate) Rc<GlQuerySetInner>);

pub struct GlQuerySetInner {
    pub(crate) state: Rc<GlDeviceState>,
    pub(crate) kind: fluxel_rhi_core::api::QueryKind,
    pub(crate) queries: Vec<crate::backend::gl::api::QueryId>,
    pub(crate) measured: std::cell::RefCell<Vec<bool>>,
}

impl core::ops::Deref for GlQuerySet {
    type Target = GlQuerySetInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GlQuerySet {
    pub(crate) fn from_inner(inner: GlQuerySetInner) -> Self {
        Self(Rc::new(inner))
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl Drop for GlQuerySetInner {
    fn drop(&mut self) {
        use crate::backend::gl::api::GlQueryObjectsApi as _;

        // Query deletion is context-affine. A failed re-entrant borrow leaves
        // the provider table to context teardown, matching other GL owners.
        let _ = self.state.with_native("drop GL query set", |provider| {
            for query in self.queries.drain(..) {
                provider.destroy_query(query)?;
            }
            Ok(())
        });
    }
}

/// Linked native compute program and the immutable ABI it was linked against.
///
/// This is intentionally separate from [`GlRasterPipeline`]: GL has one
/// current-program slot, yet a compute pipeline has neither vertex state nor
/// raster fixed state.  The encoder selects this program directly on the owner
/// context; retaining it here only keeps the native object alive across an
/// immediate pass and never stores dispatch commands.
#[derive(Clone)]
pub struct GlComputePipeline(Rc<GlComputePipelineInner>);

pub struct GlComputePipelineInner {
    pub(crate) state: Rc<GlDeviceState>,
    pub(crate) layout: GlPipelineLayout,
    pub(crate) reflection: crate::backend::gl::api::GlProgramReflection,
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) program: crate::backend::gl::api::ProgramId,
}

impl core::ops::Deref for GlComputePipeline {
    type Target = GlComputePipelineInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GlComputePipeline {
    pub(crate) fn from_inner(inner: GlComputePipelineInner) -> Self {
        Self(Rc::new(inner))
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl Drop for GlComputePipelineInner {
    fn drop(&mut self) {
        use crate::backend::gl::api::GlShaderApi as _;

        // Final release can happen while another direct verb owns the provider
        // borrow.  Failing that borrow deliberately leaves deletion to context
        // teardown; Drop must not re-enter a native table mutation.
        let _ = self
            .state
            .with_native("drop GL compute pipeline", |provider| {
                provider.destroy_program(self.program)
            });
    }
}

/// A Core binding's resource kind and the validation data needed before an
/// immediate native bind. Program reflection chooses the actual GL point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlBindingSlot {
    UniformBuffer {
        dynamic_offset: bool,
        min_size: u64,
    },
    SampledTexture {
        dimension: fluxel_rhi_core::api::TextureViewDimension,
    },
    StorageTexture {
        dimension: fluxel_rhi_core::api::TextureViewDimension,
        format: fluxel_rhi_core::api::TextureFormat,
        access: fluxel_rhi_core::api::StorageAccess,
    },
    Sampler,
}

/// One validated Core layout entry; `binding` is logical, not a GL slot.
#[derive(Clone)]
pub(crate) struct GlBindingLayoutEntry {
    pub(crate) binding: u32,
    pub(crate) slot: GlBindingSlot,
}

/// A small immutable layout for the concrete uniform-buffer plus sampled
/// texture route. It owns no GL name; program linking assigns those names.
#[derive(Clone)]
pub struct GlBindGroupLayout {
    pub(crate) state: Rc<GlDeviceState>,
    pub(crate) entries: Rc<[GlBindingLayoutEntry]>,
}

/// Resources retained by one bind group so native allocation lifetime extends
/// through every immediate raster binding that uses the group.
#[derive(Clone)]
pub(crate) enum GlBoundResource {
    UniformBuffer {
        buffer: GlBuffer,
        offset: u32,
        size: u32,
    },
    Texture(GlTextureView),
    StorageTexture(GlTextureView),
    Sampler(GlSampler),
}

/// An immutable, fully validated binding packet. Applying it issues GL calls
/// immediately; the packet is resource ownership and validation data, not a
/// deferred command list.
#[derive(Clone)]
pub struct GlBindGroup(Rc<GlBindGroupInner>);

pub struct GlBindGroupInner {
    pub(crate) layout: GlBindGroupLayout,
    pub(crate) resources: Rc<[(u32, GlBoundResource)]>,
}

impl core::ops::Deref for GlBindGroup {
    type Target = GlBindGroupInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GlBindGroup {
    pub(crate) fn from_inner(inner: GlBindGroupInner) -> Self {
        Self(Rc::new(inner))
    }
}

/// A texture subresource view. GL has no standalone view name for the 02
/// baseline, so this retains the allocation and its validated native view.
#[derive(Clone)]
pub struct GlTextureView {
    pub(crate) texture: GlTexture,
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) native: crate::backend::gl::api::GlTextureView,
}

/// One provider-owned sampler allocation shared by cloned Core handles.
#[derive(Clone)]
pub struct GlSampler(Rc<GlSamplerInner>);

pub struct GlSamplerInner {
    pub(crate) state: Rc<GlDeviceState>,
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) id: crate::backend::gl::api::SamplerId,
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub(crate) browser: crate::backend::gl::browser::BrowserSampler,
}

impl core::ops::Deref for GlSampler {
    type Target = GlSamplerInner;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GlSampler {
    pub(crate) fn from_inner(inner: GlSamplerInner) -> Self {
        Self(Rc::new(inner))
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl Drop for GlSamplerInner {
    fn drop(&mut self) {
        use crate::backend::gl::api::GlSamplerApi as _;
        // See buffer teardown: a re-entrant final release leaves the provider
        // table for context teardown rather than panicking during Drop.
        let _ = self.state.with_native("drop GL sampler", |provider| {
            provider.destroy_sampler(self.id)
        });
    }
}
/// Immutable GL pipeline ABI.
///
/// `groups` retains the exact logical layout at every Core index while
/// `native.bindings` is its flattened shader-link input. Keeping both avoids
/// treating a matching individual group as evidence that the complete
/// pipeline ABI is compatible.
#[derive(Clone)]
pub struct GlPipelineLayout {
    pub(crate) state: Rc<GlDeviceState>,
    pub(crate) groups: Rc<[GlBindGroupLayout]>,
    pub(crate) native: crate::backend::gl::api::GlPipelineLayout,
}

/// Linked program, structural vertex-array layout, and fixed raster state.
///
/// The raw GL names stay in the provider table. Clones retain one inner record
/// so a program and its VAO are deleted only after the final Core pipeline,
/// immediate encoder, and pass-local use have released them.
#[derive(Clone)]
pub struct GlRasterPipeline(Rc<GlRasterPipelineInner>);

pub struct GlRasterPipelineInner {
    pub(crate) state: Rc<GlDeviceState>,
    /// Immutable Core layout ABI used when the linked program assigned GL
    /// binding points. Raster bind groups must match this exact declaration.
    pub(crate) layout: GlPipelineLayout,
    /// Program-link evidence maps logical Core bindings to actual UBO points
    /// and combined sampler units. Immediate binds use this map directly.
    pub(crate) reflection: crate::backend::gl::api::GlProgramReflection,
    /// Structural input ABI. WebGL2 has to re-emit attribute pointers when a
    /// Core setter changes one buffer slot because those pointers retain the
    /// concrete buffer name in the VAO.
    pub(crate) vertex_layout: crate::backend::gl::api::GlVertexLayout,
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) program: crate::backend::gl::api::ProgramId,
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) vertex_array: crate::backend::gl::api::VertexArrayId,
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub(crate) browser_program: crate::backend::gl::browser::BrowserRasterProgram,
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub(crate) browser_vertex_array: crate::backend::gl::browser::BrowserVertexArray,
    pub(crate) raster: crate::backend::gl::api::GlRasterState,
}

impl core::ops::Deref for GlRasterPipeline {
    type Target = GlRasterPipelineInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GlRasterPipeline {
    pub(crate) fn from_inner(inner: GlRasterPipelineInner) -> Self {
        Self(Rc::new(inner))
    }

    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) fn native_with_dynamic_state(
        &self,
        viewport: crate::backend::gl::api::GlViewport,
        scissor: Option<crate::backend::gl::api::GlScissorRect>,
    ) -> crate::backend::gl::api::GlRasterPipeline {
        let mut state = self.raster.clone();
        state.viewport = viewport;
        state.scissor = scissor;
        crate::backend::gl::api::GlRasterPipeline {
            program: self.program,
            vertex_array: self.vertex_array,
            state,
        }
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl Drop for GlRasterPipelineInner {
    fn drop(&mut self) {
        use crate::backend::gl::api::{GlShaderApi as _, GlVertexApi as _};

        // A final release may occur while an immediate verb holds the provider
        // table. The fallible borrow in `with_native` therefore makes teardown
        // conservative: context/provider teardown retains names rather than a
        // destructor panicking or deleting a name under an active GL call.
        let _ = self
            .state
            .with_native("drop GL raster pipeline", |provider| {
                let vao = provider.destroy_vertex_array(self.vertex_array);
                let program = provider.destroy_program(self.program);
                vao.and(program)
            });
    }
}

/// A provider-owned buffer allocation plus its immutable validation descriptor.
///
/// Clones retain one allocation record. The record, rather than every wrapper,
/// owns the native deletion so views and bound packets may extend its lifetime.
#[derive(Clone)]
pub struct GlBuffer(Rc<GlBufferInner>);

/// Private-data carrier made public only because `GlBuffer` is a public Core
/// associated type. Its fields stay crate-private and never expose GL handles.
pub struct GlBufferInner {
    pub(crate) state: Rc<GlDeviceState>,
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) id: crate::backend::gl::api::BufferId,
    pub(crate) desc: crate::backend::gl::api::GlBufferDesc,
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub(crate) browser: crate::backend::gl::browser::BrowserBuffer,
}

impl core::ops::Deref for GlBuffer {
    type Target = GlBufferInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GlBuffer {
    pub(crate) fn from_inner(inner: GlBufferInner) -> Self {
        Self(Rc::new(inner))
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl Drop for GlBufferInner {
    fn drop(&mut self) {
        use crate::backend::gl::api::GlResourceApi as _;

        // Drop cannot return an error. If the platform has already declared
        // loss, the provider's tables are quarantined with the context and no
        // GL delete may be issued. Otherwise this makes the exact owner context
        // current before deleting the live name.
        // A final release can occur while an earlier provider call temporarily
        // retains an object. `with_native` uses fallible RefCell borrows, so
        // this path never panics; a re-entrant release conservatively leaves
        // the name for provider/context teardown instead of deleting it under
        // an active table borrow.
        let _ = self.state.with_native("drop GL buffer", |provider| {
            provider.destroy_buffer_resource(self.id)
        });
    }
}

/// A provider-owned texture allocation. Views retain it to preserve the native
/// texture lifetime through every view and immediate command that uses it.
#[derive(Clone)]
pub struct GlTexture(Rc<GlTextureInner>);

/// See [`GlBufferInner`]: public type shape, crate-private native data.
pub struct GlTextureInner {
    pub(crate) state: Rc<GlDeviceState>,
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) id: crate::backend::gl::api::TextureId,
    pub(crate) desc: crate::backend::gl::api::GlTextureDesc,
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub(crate) browser: crate::backend::gl::browser::BrowserTexture,
}

impl core::ops::Deref for GlTexture {
    type Target = GlTextureInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GlTexture {
    pub(crate) fn from_inner(inner: GlTextureInner) -> Self {
        Self(Rc::new(inner))
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl Drop for GlTextureInner {
    fn drop(&mut self) {
        use crate::backend::gl::api::GlResourceApi as _;

        // See GlBufferInner: context loss is deliberately conservative, while
        // a live context receives exactly one deletion at the final Rc release.
        let _ = self.state.with_native("drop GL texture", |provider| {
            provider.destroy_texture_resource(self.id)
        });
    }
}

/// Immutable shader source and provider-owned native shader identity.
#[derive(Clone)]
pub struct GlShader(Rc<GlShaderInner>);

pub struct GlShaderInner {
    pub(crate) state: Rc<GlDeviceState>,
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) id: crate::backend::gl::api::ShaderId,
    pub(crate) key: crate::backend::gl::api::GlShaderKey,
    pub(crate) source: crate::backend::gl::api::GlShaderSource,
    /// Names emitted by Naga for Core group/binding declarations. They are
    /// consumed only while linking a pipeline on this context.
    pub(crate) naga_bindings: Vec<GlNagaBindingName>,
}

#[derive(Clone)]
pub(crate) struct GlNagaBindingName {
    /// Core bind-group index. A GLSL program has no native equivalent, so this
    /// remains alongside the generated name until layout linking validates it.
    pub(crate) group: u32,
    pub(crate) binding: u32,
    pub(crate) name: String,
    /// Unit encoded in the GLSL declaration for an image uniform.
    pub(crate) image_unit: Option<u32>,
}

impl core::ops::Deref for GlShader {
    type Target = GlShaderInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GlShader {
    pub(crate) fn from_inner(inner: GlShaderInner) -> Self {
        Self(Rc::new(inner))
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
impl Drop for GlShaderInner {
    fn drop(&mut self) {
        use crate::backend::gl::api::GlShaderApi as _;

        // The provider may already be borrowed by a direct immediate verb.
        // Failing that fallible borrow leaves the name to context teardown;
        // Drop must never panic or re-enter a live native table mutation.
        let _ = self.state.with_native("drop GL shader", |provider| {
            provider.destroy_shader(self.id)
        });
    }
}
