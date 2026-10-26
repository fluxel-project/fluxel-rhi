//! Native query-object execution and result observation.
//!
//! Query operations issue directly through the current GL provider. They keep
//! query liveness and target validation private; Core controls the command
//! ordering and completion point used before a result is consumed.

use super::super::provider::{NativeGlProvider, QueryTarget};
use crate::backend::state::{ScopedRawAccess, StateEvent};

const GLES_ANY_SAMPLES_PASSED: u32 = 0x8C2F;

/// The seven counters whose stages can be produced by the current RHI.
/// Geometry and tessellation are not exposed and their result fields are zero.
pub(crate) const STATISTICS_TARGETS: [(u32, usize); 7] = [
    (0x82EE, 0),
    (0x82EF, 1),
    (0x82F0, 2),
    (0x82F6, 5),
    (0x82F7, 6),
    (0x82F4, 7),
    (0x82F5, 10),
];

fn occlusion_query_target(provider: &NativeGlProvider) -> u32 {
    if matches!(
        provider.discovery().context().profile(),
        crate::backend::api::GlFamilyProfile::Embedded { .. }
    ) {
        GLES_ANY_SAMPLES_PASSED
    } else {
        glow::SAMPLES_PASSED
    }
}
use crate::backend::api::{
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
                segments: Vec::new(),
                invalidated: false,
                target: None,
                disjoint: false,
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
        unsafe {
            self.gl.delete_query(entry.raw);
            for segment in entry.segments {
                self.gl.delete_query(segment);
            }
        };
        self.driver_error(OP)
    }

    fn query_result(&mut self, query: QueryId) -> Result<GlQueryResult, GlError> {
        use glow::HasContext as _;
        const OP: &str = "query-result";
        self.assert_ready(OP)?;
        self.query(OP, query)?;
        if self.query(OP, query)?.invalidated {
            return Ok(GlQueryResult::Unknown);
        }
        let (raw, segments, target, active) = match self.queries.get(&query) {
            Some(entry) => (
                entry.raw,
                entry.segments.clone(),
                entry.target,
                self.active_query == Some(query),
            ),
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
        for segment in &segments {
            // SAFETY: segments belong to the live inactive logical query.
            let available = unsafe {
                self.gl
                    .get_query_parameter_u32(*segment, glow::QUERY_RESULT_AVAILABLE)
            };
            if available == 0 {
                return Ok(GlQueryResult::Pending);
            }
            if available != 1 {
                return Ok(GlQueryResult::Unknown);
            }
        }
        if matches!(
            target,
            Some(QueryTarget::Timestamp | QueryTarget::TimeElapsed)
        ) && self.gles_timer.is_some()
        {
            self.observe_timer_disjoint(OP)?;
            if self.queries.get(&query).is_some_and(|entry| entry.disjoint) {
                return Ok(GlQueryResult::Disjoint);
            }
        }
        // SAFETY: the same live, inactive query was checked for availability
        // immediately above on this provider's current context. Occlusion
        // samples use the portable 32-bit GL query result; only timer domains
        // use the 64-bit query accessor.
        let value = unsafe {
            if target == Some(QueryTarget::SamplesPassed) {
                u64::from(self.gl.get_query_parameter_u32(raw, glow::QUERY_RESULT))
            } else if let Some(timer) = &self.gles_timer {
                let mut result = 0;
                (timer.query_result)(raw.0.get(), glow::QUERY_RESULT, &mut result);
                result
            } else {
                self.gl.get_query_parameter_u64(raw, glow::QUERY_RESULT)
            }
        };
        let value = if target == Some(QueryTarget::SamplesPassed) {
            segments.into_iter().fold(value, |value, segment| {
                // SAFETY: all segments were confirmed available above.
                value
                    | u64::from(unsafe {
                        self.gl.get_query_parameter_u32(segment, glow::QUERY_RESULT)
                    })
            })
        } else {
            value
        };
        Ok(GlQueryResult::Available(value))
    }
}

impl NativeGlProvider {
    pub(crate) fn invalidate_active_occlusion(&mut self) {
        if let Some(id) = self.active_query {
            if let Some(entry) = self.queries.get_mut(&id) {
                if entry.target == Some(QueryTarget::SamplesPassed) {
                    entry.invalidated = true;
                }
            }
        }
    }
    /// End the current GLES occlusion segment before internal rendering.
    pub(crate) fn suspend_occlusion_for_internal_draw(
        &mut self,
    ) -> Result<Option<glow::NativeQuery>, GlError> {
        use glow::HasContext as _;
        const OP: &str = "suspend occlusion for internal draw";
        let Some(id) = self.active_query else {
            return Ok(None);
        };
        if self.query(OP, id)?.target != Some(QueryTarget::SamplesPassed) {
            return Ok(None);
        }
        // SAFETY: this provider owns the current context and active query.
        let next = unsafe { self.gl.create_query() }.map_err(|message| GlError::Driver {
            operation: OP,
            message,
        })?;
        unsafe { self.gl.end_query(occlusion_query_target(self)) };
        self.defer_driver_error(OP)?;
        Ok(Some(next))
    }

    /// Resume the same logical measurement using a fresh native query.
    pub(crate) fn resume_occlusion_after_internal_draw(
        &mut self,
        next: Option<glow::NativeQuery>,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "resume occlusion after internal draw";
        let Some(next) = next else {
            return Ok(());
        };
        let Some(id) = self.active_query else {
            unsafe { self.gl.delete_query(next) };
            return Err(Self::validation(OP, "logical query disappeared"));
        };
        // SAFETY: the previous segment ended and this fresh query belongs to the current context.
        unsafe { self.gl.begin_query(occlusion_query_target(self), next) };
        self.defer_driver_error(OP)?;
        self.queries
            .get_mut(&id)
            .ok_or_else(|| Self::validation(OP, "logical query disappeared"))?
            .segments
            .push(next);
        Ok(())
    }

    pub(crate) fn gles_timestamp_bits(&self) -> Option<u32> {
        self.gles_timer.as_ref().map(|timer| timer.bits)
    }

    fn observe_timer_disjoint(&mut self, operation: &'static str) -> Result<(), GlError> {
        use glow::HasContext as _;
        // SAFETY: GPU_DISJOINT_EXT is legal only after the extension and its
        // entry points were proved for this current context.
        let disjoint = unsafe { self.gl.get_parameter_i32(0x8FBB) };
        self.driver_error(operation)?;
        if disjoint != 0 {
            for entry in self.queries.values_mut() {
                if matches!(
                    entry.target,
                    Some(QueryTarget::Timestamp | QueryTarget::TimeElapsed)
                ) {
                    entry.disjoint = true;
                }
            }
        }
        Ok(())
    }
    pub(crate) fn supports_pipeline_statistics(&self) -> bool {
        use crate::backend::api::GlFamilyProfile;
        use glow::HasContext as _;
        if !matches!(
            self.discovery.context().profile(),
            GlFamilyProfile::Desktop { .. }
        ) || !self
            .discovery
            .extensions()
            .is_acquired(crate::backend::api::GlKnownExtension::ArbPipelineStatisticsQuery)
        {
            return false;
        }
        let Some(get_query_iv) = self.get_query_iv else {
            return false;
        };
        for (target, _) in STATISTICS_TARGETS {
            let mut bits = 0;
            // SAFETY: loader and current context share the same owner; the
            // extension proves each target and `bits` is writable.
            unsafe { get_query_iv(target, glow::QUERY_COUNTER_BITS, &mut bits) };
            // SAFETY: the provider owns the current context.
            if unsafe { self.gl.get_error() } != glow::NO_ERROR || bits <= 0 {
                return false;
            }
        }
        true
    }

    pub(crate) fn begin_pipeline_statistics(&mut self, queries: &[QueryId]) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "begin-pipeline-statistics";
        self.assert_ready(OP)?;
        if queries.len() != STATISTICS_TARGETS.len()
            || !self.active_statistics.is_empty()
            || !self.supports_pipeline_statistics()
        {
            return Err(Self::validation(
                OP,
                "invalid or unsupported statistics query",
            ));
        }
        let raw = queries
            .iter()
            .map(|query| self.query(OP, *query).map(|entry| entry.raw))
            .collect::<Result<Vec<_>, _>>()?;
        for ((target, _), (query, raw)) in STATISTICS_TARGETS
            .into_iter()
            .zip(queries.iter().copied().zip(raw))
        {
            // SAFETY: every query is live and each target is distinct and supported.
            unsafe { self.gl.begin_query(target, raw) };
            self.queries.get_mut(&query).unwrap().target = Some(QueryTarget::Statistics(target));
            self.active_statistics.push((target, query));
            self.defer_driver_error(OP)?;
        }
        Ok(())
    }

    pub(crate) fn end_pipeline_statistics(&mut self) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "end-pipeline-statistics";
        self.assert_ready(OP)?;
        if self.active_statistics.len() != STATISTICS_TARGETS.len() {
            return Err(Self::validation(OP, "no statistics query active"));
        }
        while let Some((target, _)) = self.active_statistics.pop() {
            // SAFETY: each target was started on this current context.
            unsafe { self.gl.end_query(target) };
            self.defer_driver_error(OP)?;
        }
        Ok(())
    }

    /// Close every target already begun when a command step aborts midway.
    pub(crate) fn abort_pipeline_statistics(&mut self) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "abort-pipeline-statistics";
        self.assert_ready(OP)?;
        for (target, _) in std::mem::take(&mut self.active_statistics) {
            // SAFETY: these targets were begun by this provider on its current context.
            unsafe { self.gl.end_query(target) };
            self.defer_driver_error(OP)?;
        }
        Ok(())
    }
    /// Writes ordered query results directly to a GPU buffer when the desktop
    /// query-buffer contract is present. Returns false without mutation on
    /// profiles that need the portable CPU fallback.
    pub(crate) fn resolve_query_buffer(
        &mut self,
        queries: &[QueryId],
        destination: crate::backend::api::GlBufferRange,
    ) -> Result<bool, GlError> {
        use crate::backend::api::GlBufferUsage;
        use glow::HasContext as _;
        const OP: &str = "resolve-query-buffer";
        self.assert_ready(OP)?;
        let supported = self.supports_query_buffer();
        if !supported {
            return Ok(false);
        }
        let (buffer, desc) = self.buffer(OP, destination.buffer)?;
        destination
            .validate_for(desc)
            .map_err(|_| Self::validation(OP, "invalid destination range"))?;
        if !desc.usage.contains(GlBufferUsage::COPY_DESTINATION)
            || destination.size != queries.len() as u64 * 8
        {
            return Err(Self::validation(OP, "invalid destination usage or size"));
        }
        let mut raw = Vec::with_capacity(queries.len());
        for query in queries {
            let entry = self.query(OP, *query)?;
            if self.active_query == Some(*query) || entry.target.is_none() {
                return Err(Self::validation(OP, "query is active or unmeasured"));
            }
            raw.push(entry.raw);
        }
        let base = usize::try_from(destination.offset)
            .map_err(|_| Self::validation(OP, "offset exceeds address size"))?;
        // SAFETY: the desktop query-buffer capability and every live object,
        // result offset, and destination range were validated above. While a
        // query buffer is bound, GL interprets the pointer argument as an
        // offset, and executes each write in command order on the GPU.
        unsafe {
            self.gl.bind_buffer(glow::QUERY_BUFFER, Some(buffer));
            for (index, query) in raw.into_iter().enumerate() {
                self.gl.get_query_parameter_u64_with_offset(
                    query,
                    glow::QUERY_RESULT,
                    base + index * 8,
                );
            }
            let result = self.defer_driver_error(OP);
            self.gl.bind_buffer(glow::QUERY_BUFFER, None);
            self.context_state
                .event(StateEvent::ScopedRawAccess(ScopedRawAccess::all()));
            result?;
        }
        Ok(true)
    }

    /// Whether ordered GPU-side query resolution is available on this context.
    pub(crate) fn supports_query_buffer(&self) -> bool {
        #[cfg(test)]
        if crate::backend::rhi::spi::TEST_DISABLE_QUERY_BUFFER.with(|disabled| disabled.get()) {
            return false;
        }
        use crate::backend::api::{GlFamilyProfile, GlKnownExtension};
        match self.discovery.context().profile() {
            GlFamilyProfile::Desktop { major, minor } => {
                major > 4
                    || (major == 4 && minor >= 4)
                    || self
                        .discovery
                        .extensions()
                        .is_acquired(GlKnownExtension::ArbQueryBufferObject)
            }
            _ => false,
        }
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
        let old_segments = std::mem::take(&mut self.queries.get_mut(&query).unwrap().segments);
        let entry = self.queries.get_mut(&query).unwrap();
        entry.invalidated = false;
        entry.target = None;
        // SAFETY: the logical query is idle; old completed measurements are no longer observable.
        for segment in old_segments {
            unsafe { self.gl.delete_query(segment) };
        }
        let raw = match self.queries.get(&query) {
            Some(entry) => entry.raw,
            None => return Err(Self::validation(OP, "query disappeared")),
        };
        // SAFETY: current-context contract; the query is live and idle.
        unsafe { self.gl.begin_query(occlusion_query_target(self), raw) };
        self.defer_driver_error(OP)?;
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
        unsafe { self.gl.end_query(occlusion_query_target(self)) };
        self.defer_driver_error(OP)
    }
}

impl GlElapsedQueryApi for NativeGlProvider {
    fn begin_elapsed_query(&mut self, query: QueryId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "begin-elapsed-query";
        self.assert_ready(OP)?;
        self.require_timer_capability(OP)?;
        if self
            .gles_timer
            .as_ref()
            .is_some_and(|timer| timer.elapsed_bits == 0)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "elapsed timer counter has zero bits",
            });
        }
        self.query(OP, query)?;
        if self.gles_timer.is_some() {
            self.observe_timer_disjoint(OP)?;
        }
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
        self.defer_driver_error(OP)?;
        if let Some(entry) = self.queries.get_mut(&query) {
            entry.target = Some(QueryTarget::TimeElapsed);
            entry.disjoint = false;
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
        self.defer_driver_error(OP)
    }
}

impl GlTimestampQueryApi for NativeGlProvider {
    fn query_timestamp(&mut self, query: QueryId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "query-timestamp";
        self.assert_ready(OP)?;
        self.require_timer_capability(OP)?;
        self.query(OP, query)?;
        if self.gles_timer.is_some() {
            self.observe_timer_disjoint(OP)?;
        }
        if self.active_query.is_some() {
            return Err(Self::validation(OP, "another query is already active"));
        }
        let raw = match self.queries.get(&query) {
            Some(entry) => entry.raw,
            None => return Err(Self::validation(OP, "query disappeared")),
        };
        // SAFETY: current-context contract; a timestamp write never becomes an
        // active begin/end pair and the counter width was probed.
        unsafe {
            if let Some(timer) = &self.gles_timer {
                (timer.query_counter)(raw.0.get(), glow::TIMESTAMP);
            } else {
                self.gl.query_counter(raw, glow::TIMESTAMP);
            }
        };
        self.defer_driver_error(OP)?;
        if let Some(entry) = self.queries.get_mut(&query) {
            entry.target = Some(QueryTarget::Timestamp);
            entry.disjoint = false;
        }
        Ok(())
    }
}

impl NativeGlProvider {
    /// Timer domains execute only on the proved `TimerQuery` capability:
    /// resolved core evidence plus a nonzero `GL_QUERY_COUNTER_BITS` answer.
    fn require_timer_capability(&self, operation: &'static str) -> Result<(), GlError> {
        if self.gles_timer.is_some()
            || self
                .discovery
                .capabilities()
                .supports(crate::backend::api::GlCapability::TimerQuery)
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
