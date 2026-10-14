//! Immediate EGL context operations and lifecycle transitions.
//!
//! Calls remain on the owning thread and establish EGL currentness before
//! exposing GL dispatch. This module contains no worker or queued commands.

use super::*;

impl EglGlesContext {
    pub(crate) fn load_unmap_buffer(
        &self,
    ) -> Result<crate::backend::gl::native::context::UnmapBufferFn, GlError> {
        let address = self
            .lease
            .egl
            .get_proc_address("glUnmapBuffer")
            .map(|function| function as *const () as *const core::ffi::c_void)
            .unwrap_or(core::ptr::null());
        if address.is_null() {
            return Err(GlError::Driver {
                operation: "load glUnmapBuffer",
                message: "EGL did not expose the GLES 3.0 core entry point".into(),
            });
        }
        // SAFETY: GLES 3.0 requires this core entry point and its ABI is
        // `GLboolean glUnmapBuffer(GLenum)`; the current EGL loader resolved it.
        Ok(unsafe { core::mem::transmute(address) })
    }

    /// Returns the requested GLES profile.  Actual driver strings are verified
    /// at construction, rather than inferred from EGL configuration bits.
    pub(crate) const fn profile(&self) -> GlFamilyProfile {
        self.version.profile()
    }

    /// Returns the native owner thread captured at construction.
    pub(crate) fn owner_thread(&self) -> OwnerThreadIdentity {
        self.owner
    }

    /// Returns the lifecycle of this context/surface pair.
    pub(crate) const fn lifecycle(&self) -> GlContextLifecycle {
        self.lifecycle
    }

    /// Makes this context current on its owner thread.
    pub(crate) fn make_current(&mut self) -> Result<(), EglProviderError> {
        self.assert_owner("eglMakeCurrent")?;
        self.refuse_unusable("eglMakeCurrent")?;
        let surface = self.surface.ok_or(GlError::InvalidLifecycle {
            operation: "eglMakeCurrent",
            lifecycle: self.lifecycle,
        })?;
        self.lease
            .egl
            .make_current(
                self.lease.display,
                Some(surface),
                Some(surface),
                Some(self.context),
            )
            .map_err(|error| EglProviderError::Egl {
                operation: "eglMakeCurrent",
                error,
            })?;
        self.lifecycle = GlContextLifecycle::Active;
        Ok(())
    }

    /// Makes this context current while preserving the immediate backend's
    /// operation label in an error returned to its caller.
    pub(crate) fn make_current_for(&mut self, operation: &'static str) -> Result<(), GlError> {
        self.make_current()
            .map_err(|error| match error.into_gl_error() {
                GlError::Driver { message, .. } => GlError::Driver { operation, message },
                other => other,
            })
    }

    pub(crate) fn kind(&self) -> crate::backend::gl::native::context::NativeContextKind {
        crate::backend::gl::native::context::NativeContextKind::Embedded
    }

    pub(crate) fn is_window_surface(&self) -> bool {
        matches!(self.kind, EglSurfaceKind::Window { .. })
    }

    pub(crate) fn drawable_extent(&self) -> Option<[u32; 2]> {
        self.known_drawable_extent()
    }

    pub(crate) fn swap_buffers(&mut self) -> Result<(), GlError> {
        self.present().map_err(EglProviderError::into_gl_error)
    }

    /// Resizes an EGL-owned pbuffer, or verifies a Host-resized window surface.
    pub(crate) fn resize_drawable(&mut self, extent: [u32; 2]) -> Result<(), GlError> {
        let EglSurfaceKind::Pbuffer(_) = self.kind else {
            let actual = self.drawable_extent().ok_or(GlError::Driver {
                operation: "configure EGL window surface",
                message: "eglQuerySurface did not report a drawable extent".into(),
            })?;
            return (actual == extent).then_some(()).ok_or(GlError::Validation {
                operation: "configure EGL window surface",
                message: "the requested extent differs from eglQuerySurface".into(),
            });
        };
        self.resize(EglPbufferSize {
            width: extent[0],
            height: extent[1],
        })
        .map_err(EglProviderError::into_gl_error)
    }

    /// Presents the current window surface.  Pbuffer contexts deliberately
    /// reject presentation instead of pretending off-screen work was shown.
    pub(crate) fn present(&mut self) -> Result<(), EglProviderError> {
        self.assert_active("eglSwapBuffers")?;
        self.make_current()?;
        if matches!(self.kind, EglSurfaceKind::Pbuffer(_)) {
            return Err(GlError::Unsupported {
                operation: "eglSwapBuffers",
                reason: "an EGL pbuffer is not presentable",
            }
            .into());
        }
        let surface = self.surface.expect("active context always owns a surface");
        self.lease
            .egl
            .swap_buffers(self.lease.display, surface)
            .map_err(|error| EglProviderError::Egl {
                operation: "eglSwapBuffers",
                error,
            })
    }

    /// Recreates an owned pbuffer at `size`; native window surfaces resize with
    /// the Host window and therefore require no EGL-side resize operation.
    ///
    /// The new extent is bounded before EGL is touched at all, so a size this
    /// context cannot map leaves the existing surface intact rather than
    /// destroying it and then failing.
    pub(crate) fn resize(&mut self, size: EglPbufferSize) -> Result<(), EglProviderError> {
        self.assert_owner("eglResize")?;
        let EglSurfaceKind::Pbuffer(_) = self.kind else {
            return Ok(());
        };
        size.checked()?;
        check_drawable_extent(
            self.snapshot.limits().max_viewport_dimensions,
            [size.width, size.height],
        )?;
        self.make_current()?;
        self.suspend()?;
        let old = self.surface.expect("suspend retains the EGL surface");
        let attributes = pbuffer_attributes(size)?;
        let surface = self
            .lease
            .egl
            .create_pbuffer_surface(self.lease.display, self.config, &attributes)
            .map_err(|error| EglProviderError::Egl {
                operation: "eglCreatePbufferSurface",
                error,
            })?;
        // The new surface is adopted before the old one is released, so a
        // refused `eglCreatePbufferSurface` -- the plausible failure here --
        // leaves the context holding the surface it already had.  The earlier
        // order destroyed first and left `surface: None`, which has no recovery
        // path short of rebuilding the provider: one driver refusal and the
        // context was permanently unusable rather than merely unresized.  The
        // old surface is not current (`suspend` above detached it), so deleting
        // it after the swap is legal.
        self.surface = Some(surface);
        self.kind = EglSurfaceKind::Pbuffer(size);
        let released = self.lease.egl.destroy_surface(self.lease.display, old);
        self.resume()?;
        released.map_err(|error| EglProviderError::Egl {
            operation: "eglDestroySurface",
            error,
        })
    }

    /// Detaches the context while a Host surface is unavailable.
    pub(crate) fn suspend(&mut self) -> Result<(), EglProviderError> {
        self.assert_owner("eglMakeCurrent")?;
        // Refused for the same reason `make_current` is, and it is not a
        // technicality: suspending would rewrite `Lost` to `Suspended`, and the
        // next resume would then reach `Active` through the door this closes.
        self.refuse_unusable("eglMakeCurrent")?;
        self.detach_self_if_current()?;
        self.lifecycle = GlContextLifecycle::Suspended;
        Ok(())
    }

    /// Refuses a state change that would revive a context EGL reported lost.
    ///
    /// Loss is durable in this provider -- [`Self::context_restored`] refuses
    /// outright, because only the Host can make a new one -- and the WGL
    /// provider refuses the identical sequence, so this is the EGL provider
    /// agreeing with its own trait default (`assert_ready` answers
    /// `ContextLost` for the same state) rather than a rule of its own.  Before
    /// this, `resume_surface` reached `Active` from `Lost` and handed out a
    /// fresh acquire lease from a context that had recorded its own death.
    fn refuse_unusable(&self, operation: &'static str) -> Result<(), EglProviderError> {
        match self.lifecycle {
            GlContextLifecycle::Lost => Err(GlError::ContextLost { operation }.into()),
            GlContextLifecycle::Disposed => Err(GlError::Disposed { operation }.into()),
            _ => Ok(()),
        }
    }

    /// Reattaches the owned context/surface after suspension.
    pub(crate) fn resume(&mut self) -> Result<(), EglProviderError> {
        self.make_current()
    }

    /// Transfers the one `glow` context created during discovery.
    ///
    /// The returned context is valid only while this provider remains active on
    /// its owner thread.  Layer 1 discovery may borrow it immediately; callers
    /// must not move it to another thread or use it after suspend/dispose.
    pub(crate) fn load_glow(&mut self) -> Result<glow::Context, EglProviderError> {
        self.make_current()?;
        self.glow
            .take()
            .ok_or(GlError::InvalidLifecycle {
                operation: "load EGL glow dispatch",
                lifecycle: self.lifecycle,
            })
            .map_err(EglProviderError::Gl)
    }

    /// Returns the discovery evidence gathered when this context opened.
    ///
    /// Discovery ran once against the exact current context during construction,
    /// so this needs no repeat currentness and never re-queries -- the same
    /// contract, and for the same reason, as the WGL surface's `discover`.
    pub(crate) fn discover(&self) -> Result<&GlDiscoverySnapshot, GlError> {
        self.assert_owner("discover EGL context")?;
        if self.lifecycle == GlContextLifecycle::Disposed {
            return Err(GlError::Disposed {
                operation: "discover EGL context",
            });
        }
        Ok(&self.snapshot)
    }

    /// Copies the discovery snapshot into a handle-free fixture report.
    pub(crate) fn evidence(&self) -> EglEvidence {
        use crate::backend::gl::api::GlCapability;
        let context = self.snapshot.context();
        let limits = self.snapshot.limits();
        let mut extensions: Vec<_> = self
            .snapshot
            .extensions()
            .raw_reported_names()
            .map(str::to_owned)
            .collect();
        extensions.sort();
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
        let extent = match self.kind {
            EglSurfaceKind::Pbuffer(size) => [size.width, size.height],
            EglSurfaceKind::Window { .. } => [0, 0],
        };
        EglEvidence {
            profile: format!("{:?}", context.profile()),
            requested_version: format!("3.{}", self.version.minor()),
            observed_version: context.version().to_owned(),
            version: context.version().to_owned(),
            shading_language_version: context.shading_language_version().to_owned(),
            vendor: context.vendor().to_owned(),
            renderer: context.renderer().to_owned(),
            driver_or_browser: context.driver_or_browser().to_owned(),
            extensions,
            capabilities,
            limits: vec![
                (
                    "max-texture-size".into(),
                    limits.max_texture_size.to_string(),
                ),
                (
                    "max-renderbuffer-size".into(),
                    limits.max_renderbuffer_size.to_string(),
                ),
                (
                    "max-vertex-attributes".into(),
                    limits.max_vertex_attributes.to_string(),
                ),
                (
                    "max-viewport".into(),
                    format!(
                        "{}x{}",
                        limits.max_viewport_dimensions[0], limits.max_viewport_dimensions[1]
                    ),
                ),
                ("max-samples".into(), limits.max_samples.to_string()),
            ],
            extent,
        }
    }

    /// Runs one operation while this context is current, lending it `glow`.
    ///
    /// The counterpart of the WGL surface's `with_current`, with two differences
    /// that are facts about this provider rather than choices.
    ///
    /// It builds a `glow` context per call instead of lending out one loaded at
    /// construction, because [`Self::load_glow`] is this provider's only loading
    /// path and it needs `&mut self`.  The cost is resolving the driver's proc
    /// addresses again on each call, which a one-shot measurement does not
    /// notice; buying it back would mean caching a `glow::Context` in a struct
    /// that currently has no field for one, for a saving nothing has asked for.
    ///
    /// And it takes `&mut self`, which is the honest signature for the same
    /// reason: two of these cannot run at once over one EGL context, and the
    /// borrow checker is a better guarantee of that than a comment would be.
    pub(crate) fn with_current<T>(
        &mut self,
        operation: &'static str,
        callback: impl FnOnce(&glow::Context) -> Result<T, GlError>,
    ) -> Result<T, EglProviderError> {
        self.assert_owner(operation)?;
        let glow = self.load_glow()?;
        callback(&glow).map_err(EglProviderError::Gl)
    }

    /// Executes a real ES 3 triangle workload on the owned pbuffer and
    /// optionally returns unmodified bottom-to-top RGBA bytes from it.
    pub(crate) fn draw_evidence(
        &mut self,
        draws: u32,
        read_colour: bool,
    ) -> Result<Option<Vec<u8>>, EglProviderError> {
        let [width, height] = match self.kind {
            EglSurfaceKind::Pbuffer(size) => [size.width, size.height],
            EglSurfaceKind::Window { .. } => {
                return Err(GlError::Unsupported {
                    operation: "EGL evidence draw",
                    reason: "fixture evidence is defined only for owned pbuffers",
                }
                .into());
            }
        };
        if width == 0 || height == 0 || draws == 0 {
            return Err(GlError::Validation {
                operation: "EGL evidence draw",
                message: "evidence draw requires a nonzero extent and draw count".into(),
            }
            .into());
        }
        self.with_current("EGL evidence draw", |gl| {
            use glow::HasContext as _;
            // SAFETY: with_current has made this owned EGL context current for
            // the complete callback; all objects are deleted before it exits.
            unsafe {
                let vertex = compile_evidence_shader(gl, glow::VERTEX_SHADER, EVIDENCE_VERTEX)?;
                let fragment =
                    match compile_evidence_shader(gl, glow::FRAGMENT_SHADER, EVIDENCE_FRAGMENT) {
                        Ok(shader) => shader,
                        Err(error) => {
                            gl.delete_shader(vertex);
                            return Err(error);
                        }
                    };
                let program = gl.create_program().map_err(|message| GlError::Driver {
                    operation: "EGL evidence draw",
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
                        operation: "EGL evidence draw",
                        message,
                    });
                }
                let vao = gl
                    .create_vertex_array()
                    .map_err(|message| GlError::Driver {
                        operation: "EGL evidence draw",
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
                        operation: "EGL evidence draw",
                        message:
                            "OpenGL ES reported an error while executing the evidence workload"
                                .into(),
                    });
                }
                Ok(colour)
            }
        })
    }

    /// Explicitly tears down EGL resources and reports the first driver error.
    pub(crate) fn dispose(&mut self) -> Result<(), EglProviderError> {
        self.assert_owner("eglDestroyContext")?;
        if self.lifecycle == GlContextLifecycle::Disposed {
            return Ok(());
        }
        self.detach_self_if_current()?;
        let mut first = None;
        if let Some(surface) = self.surface.take() {
            if let Err(error) = self.lease.egl.destroy_surface(self.lease.display, surface) {
                first = Some(EglProviderError::Egl {
                    operation: "eglDestroySurface",
                    error,
                });
            }
        }
        if let Err(error) = self
            .lease
            .egl
            .destroy_context(self.lease.display, self.context)
        {
            first.get_or_insert(EglProviderError::Egl {
                operation: "eglDestroyContext",
                error,
            });
        }
        self.lifecycle = GlContextLifecycle::Disposed;
        first.map_or(Ok(()), Err)
    }

    fn assert_owner(&self, operation: &'static str) -> Result<(), GlError> {
        let actual = OwnerThreadIdentity::current();
        if actual == self.owner {
            Ok(())
        } else {
            Err(GlError::WrongThread {
                operation,
                expected: self.owner,
                actual,
            })
        }
    }

    fn assert_active(&self, operation: &'static str) -> Result<(), EglProviderError> {
        self.assert_owner(operation)?;
        match self.lifecycle {
            GlContextLifecycle::Active => Ok(()),
            GlContextLifecycle::Disposed => Err(GlError::Disposed { operation }.into()),
            lifecycle => Err(GlError::InvalidLifecycle {
                operation,
                lifecycle,
            }
            .into()),
        }
    }

    /// Detaches only if this exact display/context is current.  `eglMakeCurrent`
    /// with no context would otherwise silently detach a sibling EGL context
    /// installed by another Fluxel device on the same owner thread.
    fn detach_self_if_current(&self) -> Result<(), EglProviderError> {
        let current_context = self.lease.egl.get_current_context();
        let current_display = self.lease.egl.get_current_display();
        if current_binding_is_self(
            current_context,
            current_display,
            self.context,
            self.lease.display,
        ) {
            self.lease
                .egl
                .make_current(self.lease.display, None, None, None)
                .map_err(|error| EglProviderError::Egl {
                    operation: "eglMakeCurrent",
                    error,
                })?;
        }
        Ok(())
    }
}
