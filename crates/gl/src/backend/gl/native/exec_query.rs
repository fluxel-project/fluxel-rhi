//! Native query-object execution and result observation.
//!
//! Query operations issue directly through the current GL provider. They keep
//! query liveness and target validation private; Core controls the command
//! ordering and completion point used before a result is consumed.

use super::super::provider::{NativeGlProvider, QueryTarget};
use crate::backend::gl::api::{
    GlElapsedQueryApi, GlError, GlFamilyApi as _, GlOcclusionQueryApi, GlQueryObjectsApi,
    GlQueryResult, GlTimestampQueryApi, QueryId,
};

impl GlQueryObjectsApi for NativeGlProvider {
    fn create_query(&mut self) -> Result<QueryId, GlError> {
        use glow::HasContext as _;
        const OP: &str = "create-query";
        self.assert_ready(OP)?;
        // SAFETY: current-context contract.
        let raw = unsafe { self.gl.create_query() }.map_err(|message| GlError::Driver {
            operation: OP,
            message,
        })?;
        let slot = self.slot(OP)?;
        let id = QueryId::new(self.context_stamp(), slot, 0);
        self.queries.insert(
            id,
            super::super::provider::NativeQuery {
                generation: id.generation,
                raw,
                target: None,
            },
        );
        Ok(id)
    }

    fn destroy_query(&mut self, query: QueryId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-query";
        self.assert_ready(OP)?;
        self.query(OP, query)?;
        let entry = self
            .queries
            .remove(&query)
            .ok_or_else(|| Self::validation(OP, "query disappeared"))?;
        if self.active_query == Some(query) {
            self.active_query = None;
        }
        // SAFETY: current-context contract; liveness was checked first.
        unsafe { self.gl.delete_query(entry.raw) };
        self.driver_error(OP)
    }

    fn query_result(&mut self, query: QueryId) -> Result<GlQueryResult, GlError> {
        use glow::HasContext as _;
        const OP: &str = "query-result";
        self.assert_ready(OP)?;
        self.query(OP, query)?;
        let (raw, target, active) = match self.queries.get(&query) {
            Some(entry) => (entry.raw, entry.target, self.active_query == Some(query)),
            None => return Err(Self::validation(OP, "query disappeared")),
        };
        if target.is_none() {
            return Err(Self::validation(OP, "query has no recorded measurement"));
        }
        if active {
            return Ok(GlQueryResult::Pending);
        }
        // SAFETY: current-context contract; the query is live and inactive.
        let available = unsafe {
            self.gl
                .get_query_parameter_u32(raw, glow::QUERY_RESULT_AVAILABLE)
        };
        // A driver answer that is not exactly zero or one is indeterminate;
        // accepted-unknown keeps the observation honest.
        if available == 0 {
            return Ok(GlQueryResult::Pending);
        }
        if available != 1 {
            return Ok(GlQueryResult::Unknown);
        }
        // SAFETY: the same live, inactive query was checked for availability
        // immediately above on this provider's current context. Occlusion
        // samples use the portable 32-bit GL query result; only timer domains
        // use the 64-bit query accessor.
        let value = unsafe {
            if target == Some(QueryTarget::SamplesPassed) {
                u64::from(self.gl.get_query_parameter_u32(raw, glow::QUERY_RESULT))
            } else {
                self.gl.get_query_parameter_u64(raw, glow::QUERY_RESULT)
            }
        };
        Ok(GlQueryResult::Available(value))
    }
}

impl NativeGlProvider {
    /// Reads one completed query result through GL's blocking `QUERY_RESULT`
    /// route. This is used only by the portable query-resolve command: its
    /// completion is therefore honest even on GL profiles without query-buffer
    /// objects.
    pub(crate) fn resolve_query_result_blocking(&mut self, query: QueryId) -> Result<u64, GlError> {
        use glow::HasContext as _;
        const OP: &str = "resolve-query-result";
        self.assert_ready(OP)?;
        let entry = self.query(OP, query)?;
        if self.active_query == Some(query) {
            return Err(Self::validation(OP, "cannot resolve an active query"));
        }
        let target = self
            .queries
            .get(&query)
            .and_then(|value| value.target)
            .ok_or_else(|| Self::validation(OP, "query has no recorded measurement"))?;
        // SAFETY: the query is live, inactive, and belongs to this current
        // context. QUERY_RESULT blocks until the GPU has made it available.
        // Occlusion is specified as an unsigned integer result in core GL and
        // GLES; widening it here supplies Core's u64 resolve layout.
        Ok(unsafe {
            if target == QueryTarget::SamplesPassed {
                u64::from(
                    self.gl
                        .get_query_parameter_u32(entry.raw, glow::QUERY_RESULT),
                )
            } else {
                self.gl
                    .get_query_parameter_u64(entry.raw, glow::QUERY_RESULT)
            }
        })
    }
}

impl GlOcclusionQueryApi for NativeGlProvider {
    fn begin_occlusion_query(&mut self, query: QueryId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "begin-occlusion-query";
        self.assert_ready(OP)?;
        self.query(OP, query)?;
        if self.active_query.is_some() {
            return Err(Self::validation(OP, "another query is already active"));
        }
        let raw = match self.queries.get(&query) {
            Some(entry) => entry.raw,
            None => return Err(Self::validation(OP, "query disappeared")),
        };
        // SAFETY: current-context contract; the query is live and idle.
        unsafe { self.gl.begin_query(glow::SAMPLES_PASSED, raw) };
        self.driver_error(OP)?;
        if let Some(entry) = self.queries.get_mut(&query) {
            entry.target = Some(QueryTarget::SamplesPassed);
        }
        self.active_query = Some(query);
        Ok(())
    }

    fn end_occlusion_query(&mut self) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "end-occlusion-query";
        self.assert_ready(OP)?;
        let Some(active) = self.active_query.take() else {
            return Err(Self::validation(OP, "no occlusion query is active"));
        };
        let Some(entry) = self.queries.get(&active) else {
            return Err(Self::validation(OP, "query disappeared"));
        };
        if entry.target != Some(QueryTarget::SamplesPassed) {
            self.active_query = Some(active);
            return Err(Self::validation(
                OP,
                "the active query is not an occlusion query",
            ));
        }
        // SAFETY: current-context contract.
        unsafe { self.gl.end_query(glow::SAMPLES_PASSED) };
        self.driver_error(OP)
    }
}

impl GlElapsedQueryApi for NativeGlProvider {
    fn begin_elapsed_query(&mut self, query: QueryId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "begin-elapsed-query";
        self.assert_ready(OP)?;
        self.require_timer_capability(OP)?;
        self.query(OP, query)?;
        if self.active_query.is_some() {
            return Err(Self::validation(OP, "another query is already active"));
        }
        let raw = match self.queries.get(&query) {
            Some(entry) => entry.raw,
            None => return Err(Self::validation(OP, "query disappeared")),
        };
        // SAFETY: current-context contract; the query is live and idle, and
        // the timer capability was proved by discovery.
        unsafe { self.gl.begin_query(glow::TIME_ELAPSED, raw) };
        self.driver_error(OP)?;
        if let Some(entry) = self.queries.get_mut(&query) {
            entry.target = Some(QueryTarget::TimeElapsed);
        }
        self.active_query = Some(query);
        Ok(())
    }

    fn end_elapsed_query(&mut self) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "end-elapsed-query";
        self.assert_ready(OP)?;
        self.require_timer_capability(OP)?;
        let Some(active) = self.active_query.take() else {
            return Err(Self::validation(OP, "no elapsed query is active"));
        };
        let Some(entry) = self.queries.get(&active) else {
            return Err(Self::validation(OP, "query disappeared"));
        };
        if entry.target != Some(QueryTarget::TimeElapsed) {
            self.active_query = Some(active);
            return Err(Self::validation(
                OP,
                "the active query is not an elapsed query",
            ));
        }
        // SAFETY: current-context contract.
        unsafe { self.gl.end_query(glow::TIME_ELAPSED) };
        self.driver_error(OP)
    }
}

impl GlTimestampQueryApi for NativeGlProvider {
    fn query_timestamp(&mut self, query: QueryId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "query-timestamp";
        self.assert_ready(OP)?;
        self.require_timer_capability(OP)?;
        self.query(OP, query)?;
        if self.active_query.is_some() {
            return Err(Self::validation(OP, "another query is already active"));
        }
        let raw = match self.queries.get(&query) {
            Some(entry) => entry.raw,
            None => return Err(Self::validation(OP, "query disappeared")),
        };
        // SAFETY: current-context contract; a timestamp write never becomes an
        // active begin/end pair and the counter width was probed.
        unsafe { self.gl.query_counter(raw, glow::TIMESTAMP) };
        self.driver_error(OP)?;
        if let Some(entry) = self.queries.get_mut(&query) {
            entry.target = Some(QueryTarget::Timestamp);
        }
        Ok(())
    }
}

impl NativeGlProvider {
    /// Timer domains execute only on the proved `TimerQuery` capability:
    /// resolved core evidence plus a nonzero `GL_QUERY_COUNTER_BITS` answer.
    fn require_timer_capability(&self, operation: &'static str) -> Result<(), GlError> {
        if self
            .discovery
            .capabilities()
            .supports(crate::backend::gl::api::GlCapability::TimerQuery)
        {
            Ok(())
        } else {
            Err(GlError::Unsupported {
                operation,
                reason: "this context did not prove a nonzero timer-query counter width",
            })
        }
    }
}
