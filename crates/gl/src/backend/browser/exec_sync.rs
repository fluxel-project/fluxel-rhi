//! Browser query object lifetime and the occlusion domain. Timer measurements
//! are classified by `exec_timer.rs`, which reads them through the acquired
//! timer commands and withholds a disrupted interval; the result path here only
//! decides which of the two domains owns a recorded target. Browser queue
//! completion is owned by `BrowserOwner`, not this discovery query provider.

use super::super::api::{
    GlError, GlFamilyApi as _, GlOcclusionQueryApi, GlQueryObjectsApi, GlQueryResult, QueryId,
};
use super::discovery::WebGl2BrowserDiscovery;
use web_sys::WebGl2RenderingContext as Gl;

/// One measurement answer, if it is an exact non-negative integer count.
///
/// Every query domain shares this reading, so a counter that is fractional,
/// negative, non-finite, or absent is reported as no answer rather than being
/// coerced into a count that was never measured.
pub(super) fn measured_value(value: f64) -> Option<u64> {
    // JavaScript numbers cannot represent every integer past 2^53 - 1.
    (value.is_finite() && (0.0..=9_007_199_254_740_991.0).contains(&value) && value.fract() == 0.0)
        .then_some(value as u64)
}

impl GlQueryObjectsApi for WebGl2BrowserDiscovery {
    fn create_query(&mut self) -> Result<QueryId, GlError> {
        const OP: &str = "create-query";
        self.assert_provider_ready(OP)?;
        let raw = self
            .raw
            .create_query()
            .ok_or(GlError::OutOfMemory { operation: OP })?;
        let slot = Self::allocate_slot(&mut self.next_query_slot, OP)?;
        let id = QueryId::new(self.context_stamp(), slot, 0);
        self.queries.insert(
            slot,
            super::objects::BrowserQuery {
                generation: id.generation,
                raw,
                target: None,
                disjoint: false,
            },
        );
        Ok(id)
    }

    fn destroy_query(&mut self, query: QueryId) -> Result<(), GlError> {
        const OP: &str = "destroy-query";
        self.query(OP, query)?;
        let entry = self
            .queries
            .remove(&query.slot)
            .ok_or_else(|| Self::validation(OP, "query disappeared"))?;
        if self.active_query == Some(query.slot) {
            self.active_query = None;
        }
        self.raw.delete_query(Some(&entry.raw));
        self.driver_error(OP)
    }

    fn query_result(&mut self, query: QueryId) -> Result<GlQueryResult, GlError> {
        const OP: &str = "query-result";
        self.assert_provider_ready(OP)?;
        self.query(OP, query)?;
        let (raw, target, active) = match self.queries.get(&query.slot) {
            Some(entry) => (
                entry.raw.clone(),
                entry.target,
                self.active_query == Some(query.slot),
            ),
            None => return Err(Self::validation(OP, "query disappeared")),
        };
        if target.is_none() {
            return Err(Self::validation(OP, "query has no recorded measurement"));
        }
        if active {
            return Ok(GlQueryResult::Pending);
        }
        // A timer target is read through the extension's own accessors, which
        // also decide whether the interval may be reported at all.
        if target.is_some_and(super::exec_timer::is_timer_target) {
            if self
                .queries
                .get(&query.slot)
                .is_some_and(|entry| entry.disjoint)
            {
                return Ok(
                    match self
                        .raw
                        .get_query_parameter(&raw, Gl::QUERY_RESULT_AVAILABLE)
                        .as_bool()
                    {
                        Some(true) => GlQueryResult::Disjoint,
                        Some(false) => GlQueryResult::Pending,
                        None => GlQueryResult::Unknown,
                    },
                );
            }
            let result = self.timer_result(OP, &raw)?;
            if result == GlQueryResult::Disjoint {
                for entry in self.queries.values_mut() {
                    if entry.target.is_some_and(super::exec_timer::is_timer_target) {
                        entry.disjoint = true;
                    }
                }
            }
            return Ok(result);
        }
        let available = self
            .raw
            .get_query_parameter(&raw, Gl::QUERY_RESULT_AVAILABLE)
            .as_bool();
        let Some(available) = available else {
            return Ok(GlQueryResult::Unknown);
        };
        if !available {
            return Ok(GlQueryResult::Pending);
        }
        let answer = self.raw.get_query_parameter(&raw, Gl::QUERY_RESULT);
        let value = answer
            .as_bool()
            .map(u64::from)
            .or_else(|| answer.as_f64().and_then(measured_value));
        Ok(match value {
            Some(value) => GlQueryResult::Available(value),
            None => GlQueryResult::Unknown,
        })
    }
}

impl GlOcclusionQueryApi for WebGl2BrowserDiscovery {
    fn begin_occlusion_query(&mut self, query: QueryId) -> Result<(), GlError> {
        const OP: &str = "begin-occlusion-query";
        self.assert_provider_ready(OP)?;
        self.query(OP, query)?;
        if self.active_query.is_some() {
            return Err(Self::validation(OP, "another query is already active"));
        }
        let raw = self
            .queries
            .get(&query.slot)
            .map(|entry| entry.raw.clone())
            .ok_or_else(|| Self::validation(OP, "query disappeared"))?;
        self.raw.begin_query(Gl::ANY_SAMPLES_PASSED, &raw);
        self.driver_error(OP)?;
        if let Some(entry) = self.queries.get_mut(&query.slot) {
            entry.target = Some(super::format_map::SAMPLES_PASSED);
        }
        self.active_query = Some(query.slot);
        Ok(())
    }

    fn end_occlusion_query(&mut self) -> Result<(), GlError> {
        const OP: &str = "end-occlusion-query";
        self.assert_provider_ready(OP)?;
        let Some(active) = self.active_query.take() else {
            return Err(Self::validation(OP, "no occlusion query is active"));
        };
        self.raw.end_query(Gl::ANY_SAMPLES_PASSED);
        // `endQuery` is the command that changes the browser's active-query
        // state.  Do not commit the local transition until its error has been
        // observed: on a recoverable API failure the same query still owns the
        // context slot and a later end must be able to close it.
        if let Err(error) = self.driver_error(OP) {
            self.active_query = Some(active);
            return Err(error);
        }
        Ok(())
    }
}
