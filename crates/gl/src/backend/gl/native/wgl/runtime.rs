//! Immediate WGL operations after a context has been opened.
//!
//! Every method binds the concrete owner context on the calling thread before
//! touching GL. It does not queue or replay commands.

use super::*;

impl WglContextSurface {
    pub(crate) fn load_unmap_buffer(
        &self,
    ) -> Result<crate::backend::gl::native::context::UnmapBufferFn, GlError> {
        let address = super::load_wgl_symbol("glUnmapBuffer");
        if address.is_null() {
            return Err(GlError::Driver {
                operation: "load glUnmapBuffer",
                message: "WGL did not expose the required core entry point".into(),
            });
        }
        // SAFETY: desktop GL 1.5+ requires this core entry point and its ABI is
        // `GLboolean glUnmapBuffer(GLenum)`; the WGL loader filters invalid sentinels.
        Ok(unsafe { core::mem::transmute(address) })
    }

    /// Runs discovery or a future executor while this RHI-owned context is current.
    pub(crate) fn with_current<T>(
        &self,
        operation: &'static str,
        callback: impl FnOnce(&glow::Context) -> Result<T, GlError>,
    ) -> Result<T, GlError> {
        self.bind_current(operation)?;
        let glow = self.glow.as_ref().ok_or(GlError::InvalidLifecycle {
            operation,
            lifecycle: self.lifecycle.get(),
        })?;
        callback(glow)
    }

    /// Transfers the one dispatch table created during context opening.
    pub(crate) fn load_glow(&mut self) -> Result<glow::Context, GlError> {
        self.bind_current("load WGL glow dispatch")?;
        self.glow.take().ok_or(GlError::InvalidLifecycle {
            operation: "load WGL glow dispatch",
            lifecycle: self.lifecycle.get(),
        })
    }

    /// Returns the discovery evidence gathered when this context opened.
    ///
    /// Discovery ran once against the exact current context during
    /// construction, so it needs no repeat currentness and never re-queries.
    pub(crate) fn discover(&self) -> Result<&GlDiscoverySnapshot, GlError> {
        self.bind_current("discover WGL context")?;
        Ok(&self.snapshot)
    }

    /// Presents the RHI-owned back buffer to the Host-owned live window.
    pub(crate) fn present(&self) -> Result<(), GlError> {
        // The dispatch table moves into NativeGlProvider during adoption.
        // SwapBuffers is a WGL/GDI call and only needs this exact context and
        // drawable to be current; requiring `self.glow` would reject every
        // successfully adopted context as InvalidLifecycle.
        self.bind_current("WGL SwapBuffers")?;
        // SAFETY: bind_current checked owner-thread affinity and made this
        // provider's retained HDC/HGLRC pair current for the complete call.
        if unsafe { gdi::swap_buffers(self.hdc) } == 0 {
            Err(WglContextError::driver("WGL SwapBuffers").as_gl_error("WGL SwapBuffers"))
        } else {
            Ok(())
        }
    }

    /// Requests the optional WGL swap-control extension when the driver exposes it.
    ///
    /// `Ok(false)` means no swap-control extension exists; presentation still
    /// works with the driver's default interval.
    pub(crate) fn set_swap_interval(&self, interval: i32) -> Result<bool, GlError> {
        self.bind_current("wglSwapIntervalEXT")?;
        // Load the extension only after this provider's final core context is
        // current; WGL function availability is context/provider dependent.
        let extensions = wgl_extra::Wgl::load_with(load_wgl_symbol);
        if !extensions.SwapIntervalEXT.is_loaded() {
            return Ok(false);
        }
        // SAFETY: the optional function pointer was checked as loaded after a
        // current WGL context existed; `interval` is the WGL integer contract.
        if unsafe { extensions.SwapIntervalEXT(interval) } == 0 {
            return Err(
                WglContextError::driver("wglSwapIntervalEXT").as_gl_error("wglSwapIntervalEXT")
            );
        }
        Ok(true)
    }

    /// Records the Host window's new drawable size; no WGL swapchain exists to resize.
    ///
    /// The size is validated against the context's recorded viewport bound
    /// before any state changes, so a window the driver could not address is
    /// rejected instead of being recorded as a usable drawable.
    pub(crate) fn resize(&self, extent: [u32; 2]) -> Result<(), GlError> {
        const OP: &str = "WGL resize";
        self.assert_owner(OP)
            .map_err(|error| error.as_gl_error(OP))?;
        check_drawable_extent(self.snapshot.limits().max_viewport_dimensions, extent)
            .map_err(|error| error.as_gl_error(OP))?;
        match self.lifecycle.get() {
            GlContextLifecycle::Active | GlContextLifecycle::Suspended => {
                self.extent.set(extent);
                self.lifecycle.set(if extent.contains(&0) {
                    GlContextLifecycle::Suspended
                } else {
                    GlContextLifecycle::Active
                });
                Ok(())
            }
            lifecycle => Err(WglContextError::Lifecycle(lifecycle).as_gl_error(OP)),
        }
    }

    /// Marks the window drawable unavailable without destroying the context.
    pub(crate) fn suspend(&self) -> Result<(), GlError> {
        self.assert_owner("WGL suspend")
            .map_err(|error| error.as_gl_error("WGL suspend"))?;
        if self.lifecycle.get() == GlContextLifecycle::Active {
            self.lifecycle.set(GlContextLifecycle::Suspended);
            Ok(())
        } else {
            Err(WglContextError::Lifecycle(self.lifecycle.get()).as_gl_error("WGL suspend"))
        }
    }

    /// Reattaches a suspended context after the Host window became available.
    pub(crate) fn resume(&self) -> Result<(), GlError> {
        self.assert_owner("WGL resume")
            .map_err(|error| error.as_gl_error("WGL resume"))?;
        match self.lifecycle.get() {
            GlContextLifecycle::Suspended => {
                self.lifecycle.set(GlContextLifecycle::Active);
                Ok(())
            }
            GlContextLifecycle::Active => Ok(()),
            lifecycle => Err(WglContextError::Lifecycle(lifecycle).as_gl_error("WGL resume")),
        }
    }

    /// Returns the requested/current Host drawable extent.
    pub(crate) fn extent(&self) -> [u32; 2] {
        self.extent.get()
    }

    /// Observes the Host window rather than treating the last configuration
    /// request as proof that the compositor applied it.
    pub(crate) fn client_extent(&self) -> Option<[u32; 2]> {
        // SAFETY: WglContextSurface retains the Host-borrowed HWND; its public
        // construction contract keeps that window alive on this owner thread.
        unsafe { gdi::client_extent(self.hwnd) }
    }

    /// Returns the owner thread and generation binding used by the provider's caller.
    pub(crate) fn owner_thread(&self) -> OwnerThreadIdentity {
        self.owner_thread
    }

    /// Copies only diagnostic facts from the discovery snapshot for fixture
    /// output. It never re-queries a driver, so a report is about the same
    /// context generation that authorised lowering.
    pub(crate) fn evidence(&self) -> WglEvidence {
        use crate::backend::gl::api::GlCapability;
        let context = self.snapshot.context();
        let flags = context.flags();
        let limits = self.snapshot.limits();
        let mut extensions: Vec<_> = self
            .snapshot
            .extensions()
            .raw_reported_names()
            .map(str::to_owned)
            .collect();
        extensions.sort();
        let mut typed_extensions: Vec<_> = extensions
            .iter()
            .filter_map(|raw| {
                let known = crate::backend::gl::api::GlKnownExtension::from_raw_name(raw)?;
                let provenance = self.snapshot.extensions().provenance(known)?;
                Some((known.raw_name().to_owned(), format!("{provenance:?}")))
            })
            .collect();
        typed_extensions.sort();
        let mut other_flags: Vec<_> = flags.other.iter().cloned().collect();
        other_flags.sort();
        let capabilities = [
            ("compute", GlCapability::Compute),
            ("storage-buffer", GlCapability::StorageBuffer),
            ("storage-image", GlCapability::StorageImage),
            ("indirect-draw", GlCapability::IndirectDraw),
            ("indirect-dispatch", GlCapability::IndirectDispatch),
            ("multi-draw", GlCapability::MultiDraw),
            ("multi-draw-indirect", GlCapability::MultiDrawIndirect),
            ("multiview", GlCapability::Multiview),
            ("timer-query", GlCapability::TimerQuery),
        ]
        .into_iter()
        .map(|(name, capability)| {
            (
                name.to_owned(),
                self.snapshot.capabilities().supports(capability),
            )
        })
        .collect();
        WglEvidence {
            profile: format!("{:?}", context.profile()),
            version: context.version().to_owned(),
            shading_language_version: context.shading_language_version().to_owned(),
            vendor: context.vendor().to_owned(),
            renderer: context.renderer().to_owned(),
            driver_or_browser: context.driver_or_browser().to_owned(),
            debug: flags.debug,
            forward_compatible: flags.forward_compatible,
            robust_access: flags.robust_access,
            no_error: flags.no_error,
            other_flags,
            extensions,
            typed_extensions,
            limits: vec![
                (
                    "max_texture_size".into(),
                    limits.max_texture_size.to_string(),
                ),
                (
                    "max_renderbuffer_size".into(),
                    limits.max_renderbuffer_size.to_string(),
                ),
                (
                    "max_vertex_attributes".into(),
                    limits.max_vertex_attributes.to_string(),
                ),
                (
                    "max_viewport".into(),
                    format!(
                        "{}x{}",
                        limits.max_viewport_dimensions[0], limits.max_viewport_dimensions[1]
                    ),
                ),
                ("max_samples".into(), limits.max_samples.to_string()),
                (
                    "uniform_buffer_offset_alignment".into(),
                    limits.uniform_buffer_offset_alignment.to_string(),
                ),
                (
                    "storage_buffer_offset_alignment".into(),
                    limits.storage_buffer_offset_alignment.to_string(),
                ),
                (
                    "max_multiview_view_count".into(),
                    self.snapshot.max_multiview_view_count().to_string(),
                ),
            ],
            capabilities,
            surface_facts: format!("{:?}", self.snapshot.surface_facts()),
            drawable_extent: self.extent(),
        }
    }

    /// Executes a real core-profile triangle workload against the default WGL
    /// framebuffer.  The fixture uses this narrow path to prove the context,
    /// program compiler, vertex path, rasterizer and readback route together;
    /// regular RHI submission remains on the v13 execution driver seam.
    pub(crate) fn draw_evidence(
        &self,
        draws: u32,
        read_colour: bool,
    ) -> Result<Option<Vec<u8>>, GlError> {
        use glow::HasContext as _;
        const OP: &str = "WGL evidence draw";
        let [width, height] = self.extent();
        if width == 0 || height == 0 || draws == 0 {
            return Err(GlError::Validation {
                operation: OP,
                message: "evidence draw requires a nonzero extent and draw count".into(),
            });
        }
        // SAFETY: `with_current` made this exact owned WGL context current for
        // the closure. Every shader/program name below is created inside the
        // closure and deleted on each local failure path before it can escape.
        self.with_current(OP, |gl| unsafe {
            let vertex = compile_evidence_shader(gl, glow::VERTEX_SHADER, EVIDENCE_VERTEX, OP)?;
            let fragment =
                match compile_evidence_shader(gl, glow::FRAGMENT_SHADER, EVIDENCE_FRAGMENT, OP) {
                    Ok(shader) => shader,
                    Err(error) => {
                        gl.delete_shader(vertex);
                        return Err(error);
                    }
                };
            let program = gl.create_program().map_err(|message| GlError::Driver {
                operation: OP,
                message,
            })?;
            gl.attach_shader(program, vertex);
            gl.attach_shader(program, fragment);
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                let message = gl.get_program_info_log(program);
                gl.delete_program(program);
                gl.delete_shader(vertex);
                gl.delete_shader(fragment);
                return Err(GlError::Driver {
                    operation: OP,
                    message,
                });
            }
            let vao = gl
                .create_vertex_array()
                .map_err(|message| GlError::Driver {
                    operation: OP,
                    message,
                })?;
            gl.bind_vertex_array(Some(vao));
            gl.use_program(Some(program));
            gl.viewport(0, 0, width as i32, height as i32);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            for _ in 0..draws {
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
            }
            gl.finish();
            let colour = if read_colour {
                let mut bytes = vec![0_u8; width as usize * height as usize * 4];
                gl.read_pixels(
                    0,
                    0,
                    width as i32,
                    height as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelPackData::Slice(Some(&mut bytes)),
                );
                Some(bytes)
            } else {
                None
            };
            gl.bind_vertex_array(None);
            gl.use_program(None);
            gl.delete_vertex_array(vao);
            gl.delete_program(program);
            gl.delete_shader(vertex);
            gl.delete_shader(fragment);
            if gl.get_error() != glow::NO_ERROR {
                return Err(GlError::Driver {
                    operation: OP,
                    message: "OpenGL reported an error while executing the evidence workload"
                        .into(),
                });
            }
            Ok(colour)
        })
    }

    fn assert_owner(&self, _operation: &'static str) -> Result<(), WglContextError> {
        let actual = OwnerThreadIdentity::current();
        if actual == self.owner_thread {
            Ok(())
        } else {
            Err(WglContextError::WrongThread {
                expected: self.owner_thread,
                actual,
            })
        }
    }

    fn assert_ready(&self, operation: &'static str) -> Result<(), WglContextError> {
        self.assert_owner(operation)?;
        match self.lifecycle.get() {
            GlContextLifecycle::Active => Ok(()),
            lifecycle => Err(WglContextError::Lifecycle(lifecycle)),
        }
    }

    /// Binds this exact RHI-owned context for every GL/WGL executable seam.
    ///
    /// This intentionally does not trust the last RHI call, another RHI
    /// context, or foreign WGL code to have preserved currentness.
    pub(super) fn bind_current(&self, operation: &'static str) -> Result<(), GlError> {
        self.assert_ready(operation)
            .map_err(|error| error.as_gl_error(operation))?;
        // SAFETY: affinity was checked and this provider owns both handles.
        // Calling MakeCurrent on each seam is the actual cross-context arbiter.
        if unsafe { wgl::MakeCurrent(self.hdc, self.hglrc) } == 0 {
            return Err(WglContextError::driver(operation).as_gl_error(operation));
        }
        RHI_CURRENT_WGL.with(|arbiter| arbiter.record(self.hglrc));
        Ok(())
    }
}
