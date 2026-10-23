//! Immediate browser query operations for Core's completion-gated resolve.
use super::owner::BrowserOwner;
use crate::backend::gl::api::{
    GlError, GlOcclusionQueryApi as _, GlQueryObjectsApi as _, GlQueryResult,
    GlTimestampQueryApi as _, QueryId,
};
use fluxel_rhi_core::api::{Error, Result};

impl BrowserOwner {
    fn query_context<T>(
        &self,
        f: impl FnOnce(&mut super::discovery::WebGl2BrowserDiscovery) -> std::result::Result<T, GlError>,
    ) -> Result<T> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| Error::InvalidState)?;
        state.ensure_active()?;
        f(&mut state.context).map_err(|error| match error {
            GlError::ContextLost { .. } => Error::DeviceLost,
            GlError::OutOfMemory { .. } => Error::OutOfMemory,
            other => Error::Backend(format!("{other:?}")),
        })
    }
    pub(crate) fn create_core_queries(&self, count: u32) -> Result<Vec<QueryId>> {
        self.query_context(|context| {
            let mut queries = Vec::new();
            for _ in 0..count {
                match context.create_query() {
                    Ok(query) => queries.push(query),
                    Err(error) => {
                        for query in queries {
                            let _ = context.destroy_query(query);
                        }
                        return Err(error);
                    }
                }
            }
            Ok(queries)
        })
    }
    pub(crate) fn destroy_core_queries(&self, queries: &[QueryId]) {
        let _ = self.query_context(|context| {
            for query in queries {
                context.destroy_query(*query)?;
            }
            Ok(())
        });
    }
    pub(crate) fn begin_core_occlusion(&self, query: QueryId) -> Result<()> {
        self.query_context(|context| context.begin_occlusion_query(query))
    }
    pub(crate) fn end_core_occlusion(&self) -> Result<()> {
        self.query_context(|context| context.end_occlusion_query())
    }
    pub(crate) fn write_core_timestamp(&self, query: QueryId) -> Result<()> {
        self.query_context(|context| context.query_timestamp(query))
    }
    pub(crate) fn core_query_results(&self, queries: &[QueryId]) -> Result<Vec<u64>> {
        let answers = self.query_context(|context| {
            queries
                .iter()
                .map(|query| context.query_result(*query))
                .collect::<std::result::Result<Vec<_>, _>>()
        })?;
        if answers
            .iter()
            .any(|answer| matches!(answer, GlQueryResult::Pending | GlQueryResult::Unknown))
        {
            return Err(Error::InvalidState);
        }
        // A later slot can consume a context-wide disjoint flag and invalidate
        // earlier slots. Re-read only then, so each slot retains its own epoch.
        let answers = if answers
            .iter()
            .any(|answer| matches!(answer, GlQueryResult::Disjoint))
        {
            self.query_context(|context| {
                queries
                    .iter()
                    .map(|query| context.query_result(*query))
                    .collect::<std::result::Result<Vec<_>, _>>()
            })?
        } else {
            answers
        };
        answers
            .into_iter()
            .map(|answer| match answer {
                GlQueryResult::Available(value) => Ok(value),
                GlQueryResult::Disjoint => Ok(u64::MAX),
                _ => Err(Error::InvalidState),
            })
            .collect()
    }
}
