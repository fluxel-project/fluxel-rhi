//! Current-context discovery, native entry-point loading, and provider assembly.
//!
//! This module constructs the provider once; it does not record commands or
//! plan resource use. The platform owner retains context currentness.
use super::*;

#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
impl NativeGlProvider {
    /// # Safety
    ///
    /// Same as [`discover_current_glow`]: the caller keeps `gl` current and
    /// exclusively owned by this thread for the provider's entire lifetime.
    pub(crate) unsafe fn from_current(
        gl: glow::Context,
        stamp: ContextStamp,
        unmap_buffer: super::super::context::UnmapBufferFn,
        proc_loader: impl Fn(&str) -> *const core::ffi::c_void,
    ) -> Result<Self, NativeDiscoveryError> {
        // SAFETY: forwarded from this constructor's current-context contract.
        let discovery = unsafe {
            super::super::discovery::discover_current_glow_with_loader(&gl, stamp, &proc_loader)
        }?;
        // SAFETY: the context is current and proc_loader resolves its entry points.
        let query_ptr = proc_loader("glGetQueryiv");
        let texture_sub_image_read = proc_loader("glGetTextureSubImage");
        let tex_image_read = proc_loader("glGetTexImage");
        let compressed_2d = proc_loader("glCompressedTexImage2D");
        let get_compressed = proc_loader("glGetCompressedTexImage");
        let compressed_3d = proc_loader("glCompressedTexImage3D");
        let texture_sub_image_1d = proc_loader("glTexSubImage1D");
        let framebuffer_texture_1d = proc_loader("glFramebufferTexture1D");
        let copy_tex_sub_image_1d = proc_loader("glCopyTexSubImage1D");
        let texture_storage_entries =
            !proc_loader("glTexStorage2D").is_null() && !proc_loader("glTexStorage3D").is_null();
        // SAFETY: these core symbols have the GL-defined ABI and belong to
        // this exact current context; null disables the mutable fallback.
        let compressed_image_2d = (!compressed_2d.is_null()).then(|| unsafe {
            std::mem::transmute::<
                *const core::ffi::c_void,
                unsafe extern "system" fn(
                    u32,
                    i32,
                    u32,
                    i32,
                    i32,
                    i32,
                    i32,
                    *const core::ffi::c_void,
                ),
            >(compressed_2d)
        });
        let get_compressed_image = (!get_compressed.is_null()).then(|| unsafe {
            std::mem::transmute::<
                *const core::ffi::c_void,
                unsafe extern "system" fn(u32, i32, *mut core::ffi::c_void),
            >(get_compressed)
        });
        let compressed_image_3d = (!compressed_3d.is_null()).then(|| unsafe {
            std::mem::transmute::<
                *const core::ffi::c_void,
                unsafe extern "system" fn(
                    u32,
                    i32,
                    u32,
                    i32,
                    i32,
                    i32,
                    i32,
                    i32,
                    *const core::ffi::c_void,
                ),
            >(compressed_3d)
        });
        let texture_sub_image_1d = (!texture_sub_image_1d.is_null()).then(|| unsafe {
            std::mem::transmute::<
                *const core::ffi::c_void,
                unsafe extern "system" fn(u32, i32, i32, i32, u32, u32, *const core::ffi::c_void),
            >(texture_sub_image_1d)
        });
        let framebuffer_texture_1d = (!framebuffer_texture_1d.is_null()).then(|| unsafe {
            std::mem::transmute::<
                *const core::ffi::c_void,
                unsafe extern "system" fn(u32, u32, u32, u32, i32),
            >(framebuffer_texture_1d)
        });
        let copy_tex_sub_image_1d = (!copy_tex_sub_image_1d.is_null()).then(|| unsafe {
            std::mem::transmute::<
                *const core::ffi::c_void,
                unsafe extern "system" fn(u32, i32, i32, i32, i32, i32),
            >(copy_tex_sub_image_1d)
        });
        // SAFETY: GL specifies this signature for glGetQueryiv and the pointer
        // belongs to the same current context as the provider.
        let get_query_iv = (!query_ptr.is_null()).then(|| unsafe {
            std::mem::transmute::<*const core::ffi::c_void, super::super::probes::GetQueryivFn>(
                query_ptr,
            )
        });
        let mut gles_timer = None;
        if matches!(
            discovery.context().profile(),
            crate::backend::api::GlFamilyProfile::Embedded { .. }
        ) && crate::backend::api::GlKnownExtension::ExtDisjointTimerQuery
            .is_legal_for(discovery.context().profile())
            && discovery
                .extensions()
                .provenance(crate::backend::api::GlKnownExtension::ExtDisjointTimerQuery)
                .is_some()
        {
            let bits_ptr = proc_loader("glGetQueryivEXT");
            let counter_ptr = proc_loader("glQueryCounterEXT");
            let result_ptr = proc_loader("glGetQueryObjectui64vEXT");
            if !bits_ptr.is_null() && !counter_ptr.is_null() && !result_ptr.is_null() {
                // SAFETY: these EXT symbols have the registry-defined signatures
                // and belong to this current context.
                let get_bits = unsafe {
                    std::mem::transmute::<
                        *const core::ffi::c_void,
                        super::super::probes::GetQueryivFn,
                    >(bits_ptr)
                };
                let query_counter = unsafe {
                    std::mem::transmute::<
                        *const core::ffi::c_void,
                        unsafe extern "system" fn(u32, u32),
                    >(counter_ptr)
                };
                let query_result = unsafe {
                    std::mem::transmute::<
                        *const core::ffi::c_void,
                        unsafe extern "system" fn(u32, u32, *mut u64),
                    >(result_ptr)
                };
                let mut bits = 0_i32;
                // SAFETY: the EXT target is valid and `bits` is writable.
                unsafe { get_bits(glow::TIMESTAMP, glow::QUERY_COUNTER_BITS, &mut bits) };
                use glow::HasContext as _;
                // SAFETY: this context remains current during discovery.
                if unsafe { gl.get_error() } == glow::NO_ERROR && (1..=64).contains(&bits) {
                    let mut elapsed_bits = 0_i32;
                    // SAFETY: the extension also defines the elapsed target.
                    unsafe {
                        get_bits(
                            glow::TIME_ELAPSED,
                            glow::QUERY_COUNTER_BITS,
                            &mut elapsed_bits,
                        )
                    };
                    // SAFETY: the same EGL context remains current.
                    if unsafe { gl.get_error() } != glow::NO_ERROR {
                        elapsed_bits = 0;
                    }
                    gles_timer = Some(GlesTimerFunctions {
                        query_counter,
                        query_result,
                        bits: bits as u32,
                        elapsed_bits: u32::try_from(elapsed_bits).unwrap_or(0),
                    });
                }
            }
        }
        let indirect_count = unsafe {
            super::super::indirect_count::IndirectCountFns::load(&discovery, &proc_loader)
        };
        // SAFETY: the proc loader belongs to this current context.
        let multi_draw_indirect = unsafe {
            super::super::multi_draw_indirect::MultiDrawIndirectFns::load(&discovery, &proc_loader)
        };
        let multiview =
            unsafe { super::super::multiview::NativeMultiview::load(&discovery, &proc_loader) };
        let mut provider = Self::assemble(gl, discovery, unmap_buffer);
        // SAFETY: these symbols are resolved by the current WGL/EGL context.
        provider.vertex_functions =
            unsafe { super::super::buffer_abi::NativeVertexFunctions::load(&proc_loader) };
        // SAFETY: these names are resolved by the same current WGL/EGL context.
        provider.draw_functions =
            unsafe { super::super::draw_abi::NativeDrawFunctions::load(&proc_loader) };
        // SAFETY: the loader belongs to this current, thread-affine context;
        // the typed table uses the native pointer-width buffer ABI.
        provider.buffer_functions =
            unsafe { super::super::buffer_abi::NativeBufferFunctions::load(&proc_loader) };
        // SAFETY: optional entries belong to this exact current desktop
        // context, whose version/extension capability is checked by the loader.
        provider.spirv =
            unsafe { super::super::spirv::NativeSpirvFns::load(&provider.discovery, &proc_loader) };
        let extensions = provider.discovery.extensions();
        let storage = super::super::buffer_mapping::storage_entry(
            provider.discovery.context().profile(),
            extensions
                .raw_reported_names()
                .any(|name| name == "GL_ARB_buffer_storage"),
            extensions
                .raw_reported_names()
                .any(|name| name == "GL_EXT_buffer_storage"),
            &proc_loader,
        );
        if let Some(storage) = storage {
            // SAFETY: the loaded entry has the GL BufferStorage ABI and the
            // feature is supported by this current context.
            provider.buffer_storage = Some(unsafe { core::mem::transmute(storage) });
        }
        if provider
            .discovery
            .capabilities()
            .supports(crate::backend::api::GlCapability::Compute)
            && provider
                .discovery
                .capabilities()
                .supports(crate::backend::api::GlCapability::StorageImage)
        {
            let desktop = matches!(
                provider.discovery.context().profile(),
                crate::backend::api::GlFamilyProfile::Desktop { .. }
            );
            provider.storage_3d_nonzero_mip = super::super::probes::GlowProbes::new(
                &provider.gl,
                None,
                [false; 3],
                [core::ptr::null(); 3],
                None,
                false,
                false,
                false,
                None,
                Default::default(),
                None,
            )
            .storage_3d_nonzero_mip(desktop)?;
        }
        // SAFETY: these names are resolved for the provider's current context.
        provider.multisample_array =
            unsafe { super::super::multisample_array::MultisampleArrayFns::load(&proc_loader) }
                .select(
                    provider.discovery.context().profile(),
                    provider.discovery.extensions(),
                );
        provider.multiview = multiview;
        if matches!(
            provider.discovery.context().profile(),
            crate::backend::api::GlFamilyProfile::Desktop { .. }
        ) {
            use glow::HasContext as _;
            // SAFETY: the provider owns this current GL 4+ context; seamless
            // cube filtering is core on desktop and GLES handles it implicitly.
            unsafe { provider.gl.enable(glow::TEXTURE_CUBE_MAP_SEAMLESS) };
            provider
                .driver_error("initialize seamless cube filtering")
                .map_err(|_| {
                    NativeDiscoveryError::QueryFailed("initialize seamless cube filtering")
                })?;
        }
        provider.indirect_count = indirect_count;
        provider.multi_draw_indirect = multi_draw_indirect;
        // SAFETY: the loader and provider belong to the same current context.
        let profile = provider.discovery.context().profile();
        let ext = provider.discovery.extensions();
        let names: &[&str] = if profile.has_core(
            crate::backend::api::GlVersion::new(4, 0),
            crate::backend::api::GlVersion::new(3, 2),
        ) {
            &["glDrawElementsInstancedBaseVertex"]
        } else if ext
            .raw_reported_names()
            .any(|name| name == "GL_OES_draw_elements_base_vertex")
        {
            &["glDrawElementsInstancedBaseVertexOES"]
        } else if ext
            .raw_reported_names()
            .any(|name| name == "GL_EXT_draw_elements_base_vertex")
        {
            &["glDrawElementsInstancedBaseVertexEXT"]
        } else {
            &[]
        };
        provider.base_vertex_draw = names
            .iter()
            .map(|name| proc_loader(name))
            .find(|ptr| !ptr.is_null())
            .map(|ptr| {
                // SAFETY: symbol is resolved from the current context for the GL ABI above.
                unsafe { core::mem::transmute(ptr) }
            });
        provider.get_query_iv = get_query_iv;
        // SAFETY: discovery and symbols refer to this current native context.
        provider.clear_texture_entry =
            unsafe { super::super::clear_texture_direct::load(&provider.discovery, &proc_loader) };
        // SAFETY: the loader belongs to the current native context.
        provider.rasterization = unsafe {
            super::super::rasterization::NativeRasterization::load(
                &provider.discovery,
                &proc_loader,
            )
        };
        let profile = provider.discovery.context().profile();
        let extensions = provider.discovery.extensions();
        let clamp_name = if matches!(profile, crate::backend::api::GlFamilyProfile::Desktop { major, minor } if (major, minor) >= (4, 6))
            || extensions
                .raw_reported_names()
                .any(|name| name == "GL_ARB_polygon_offset_clamp")
        {
            Some("glPolygonOffsetClamp")
        } else if extensions
            .raw_reported_names()
            .any(|name| name == "GL_EXT_polygon_offset_clamp")
        {
            Some("glPolygonOffsetClampEXT")
        } else {
            None
        };
        provider.polygon_offset_clamp = clamp_name
            .map(&proc_loader)
            .filter(|entry| !entry.is_null())
            .map(|entry| {
                // SAFETY: both core and EXT variants use this exact GLfloat ABI,
                // and the symbol belongs to the current context's supported API.
                unsafe { core::mem::transmute(entry) }
            });
        let supports_sample_mask = match provider.discovery.context().profile() {
            crate::backend::api::GlFamilyProfile::Desktop { .. } => true,
            crate::backend::api::GlFamilyProfile::Embedded { major, minor } => {
                (major, minor) >= (3, 1)
            }
            _ => false,
        };
        if supports_sample_mask {
            let entry = proc_loader("glSampleMaski");
            if !entry.is_null() {
                // SAFETY: the current-context loader returned the core GL/GLES
                // symbol with the GLuint word-index and GLbitfield mask ABI.
                provider.sample_mask_entry = Some(unsafe { core::mem::transmute(entry) });
            }
        }
        let clamp_color = proc_loader("glClampColor");
        if !clamp_color.is_null() {
            // SAFETY: this current-context symbol has the GL enum, enum ABI.
            provider.clamp_read_color = Some(unsafe {
                core::mem::transmute::<*const core::ffi::c_void, unsafe extern "system" fn(u32, u32)>(
                    clamp_color,
                )
            });
        }
        provider.compressed_image_2d = compressed_image_2d;
        provider.get_compressed_image = get_compressed_image;
        provider.compressed_image_3d = compressed_image_3d;
        provider.texture_sub_image_1d = texture_sub_image_1d;
        // SAFETY: GL specifies this entry point's ABI and the loader belongs to this context.
        // SAFETY: GL specifies the entry point ABI and this loader belongs to the current context.
        provider.get_tex_image = (!tex_image_read.is_null()).then(|| unsafe {
            std::mem::transmute::<
                *const core::ffi::c_void,
                super::super::exec_texture_sub_image_readback::GetTexImageFn,
            >(tex_image_read)
        });
        provider.get_texture_sub_image = (!texture_sub_image_read.is_null()).then(|| unsafe {
            std::mem::transmute::<
                *const core::ffi::c_void,
                super::super::exec_texture_sub_image_readback::GetTextureSubImageFn,
            >(texture_sub_image_read)
        });
        provider.framebuffer_texture_1d = framebuffer_texture_1d;
        provider.copy_tex_sub_image_1d = copy_tex_sub_image_1d;
        provider.texture_storage_entries = texture_storage_entries;
        provider.texture_view_entry =
            unsafe { super::super::texture_view::load(&provider.discovery, &proc_loader) };
        provider.sampler_border = unsafe {
            super::super::sampler_border::SamplerBorderFns::load(&provider.discovery, &proc_loader)
        };
        provider.vertex_binding_entries = [
            "glBindVertexBuffer",
            "glVertexAttribFormat",
            "glVertexAttribIFormat",
            "glVertexAttribBinding",
            "glVertexBindingDivisor",
        ]
        .iter()
        .all(|name| !proc_loader(name).is_null());
        // SAFETY: loader resolves symbols for this current native context.
        provider.program_binary_entries =
            unsafe { super::super::program_cache::ProgramBinaryFns::load(&proc_loader) };
        // SAFETY: proc_loader resolves entries from the provider context.
        provider.storage_block_entries =
            unsafe { super::super::program_resources::StorageBlockEntries::load(&proc_loader) };
        let copy_name = match provider.discovery.context().profile() {
            crate::backend::api::GlFamilyProfile::Desktop { .. }
                if provider
                    .discovery
                    .extensions()
                    .raw_reported_names()
                    .any(|name| name == "GL_ARB_copy_image") =>
            {
                Some("glCopyImageSubData")
            }
            crate::backend::api::GlFamilyProfile::Embedded { .. }
                if provider
                    .discovery
                    .extensions()
                    .raw_reported_names()
                    .any(|name| name == "GL_EXT_copy_image")
                    && !proc_loader("glCopyImageSubDataEXT").is_null() =>
            {
                Some("glCopyImageSubDataEXT")
            }
            crate::backend::api::GlFamilyProfile::Embedded { .. }
                if provider
                    .discovery
                    .extensions()
                    .raw_reported_names()
                    .any(|name| name == "GL_OES_copy_image") =>
            {
                Some("glCopyImageSubDataOES")
            }
            _ => None,
        };
        if let Some(name) = copy_name {
            let ptr = proc_loader(name);
            if !ptr.is_null() {
                // SAFETY: all three extension spellings have this exact GL ABI.
                provider.copy_image_extension = Some(unsafe {
                    std::mem::transmute::<
                        *const core::ffi::c_void,
                        unsafe extern "system" fn(
                            u32,
                            u32,
                            i32,
                            i32,
                            i32,
                            i32,
                            u32,
                            u32,
                            i32,
                            i32,
                            i32,
                            i32,
                            i32,
                            i32,
                            i32,
                        ),
                    >(ptr)
                });
            }
        }
        provider.gles_timer = gles_timer;
        Ok(provider)
    }

    /// Assembles the provider over an already-collected discovery snapshot.
    ///
    /// The caller must guarantee `gl` is the exact context that produced
    /// `discovery` and that it is current on the calling thread.
    ///
    /// # Safety
    ///
    /// Same as [`Self::from_current`].
    pub(crate) unsafe fn from_discovered(
        gl: glow::Context,
        discovery: GlDiscoverySnapshot,
        unmap_buffer: super::super::context::UnmapBufferFn,
    ) -> Self {
        Self::assemble(gl, discovery, unmap_buffer)
    }

    fn assemble(
        gl: glow::Context,
        discovery: GlDiscoverySnapshot,
        unmap_buffer: super::super::context::UnmapBufferFn,
    ) -> Self {
        Self {
            gl: std::rc::Rc::new(gl),
            unmap_buffer,
            buffer_functions: None,
            vertex_functions: None,
            draw_functions: Default::default(),
            spirv: None,
            buffer_storage: None,
            persistent_buffers: BTreeMap::new(),
            program_resource: None,
            storage_block_entries: None,
            indexed_blend: Default::default(),
            indirect_count: None,
            multi_draw_indirect: None,
            indirect_count_utility: None,
            indirect_count_scratch: Default::default(),
            multiview: None,
            multisample_array: None,
            storage_3d_nonzero_mip: false,
            sample_mask_entry: None,
            polygon_offset_clamp: None,
            rasterization: Default::default(),
            clear_texture_entry: None,
            base_vertex_draw: None,
            get_query_iv: None,
            get_tex_image: None,
            get_texture_sub_image: None,
            clamp_read_color: None,
            compressed_image_2d: None,
            get_compressed_image: None,
            compressed_image_3d: None,
            texture_sub_image_1d: None,
            framebuffer_texture_1d: None,
            copy_tex_sub_image_1d: None,
            texture_storage_entries: false,
            texture_view_entry: None,
            vertex_binding_entries: false,
            program_binary_entries: None,
            cache_retrievable_hint: false,
            copy_image_extension: None,
            gles_timer: None,
            active_statistics: Vec::new(),
            discovery,
            lifecycle: GlContextLifecycle::Active,
            owner: OwnerThreadIdentity::current(),
            next_slot: 0,
            buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
            renderbuffers: BTreeMap::new(),
            samplers: BTreeMap::new(),
            dummy_textures: BTreeMap::new(),
            dummy_samplers: BTreeMap::new(),
            dummy_storage_images: Vec::new(),
            dummy_uniform_buffers: BTreeMap::new(),
            dummy_storage_buffers: BTreeMap::new(),
            sampler_border: None,
            texture_unit_classes: BTreeMap::new(),
            texture_unit_samplers: BTreeMap::new(),
            shaders: BTreeMap::new(),
            programs: BTreeMap::new(),
            vertex_arrays: BTreeMap::new(),
            framebuffers: BTreeMap::new(),
            queries: BTreeMap::new(),
            syncs: BTreeMap::new(),
            fences: GlFenceLeaseBook::default(),
            surface: GlSurfaceLeaseBook::new(),
            surface_suspended: false,
            // The Host has reported no extent yet, so the executor starts with
            // none: a provider that guessed a size here would hand out a lease
            // for a drawable it has never been told about.
            surface_extent: None,
            pass: None,
            raster: None,
            current_program: None,
            bound_vertex_array: None,
            active_query: None,
            active_compute_program: None,
            pixel_store: GlPixelStoreState::DEFAULT,
            context_state: ContextState::new(ExecutionMode::Optimized),
            pipeline_blocks: RasterPipelineBlockInterner::new(),
        }
    }
}
