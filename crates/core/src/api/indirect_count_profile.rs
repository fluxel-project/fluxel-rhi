//! Immutable lowering facts for planning GPU-counted indirect draws.

use super::{Buffer, Device, Error, RasterBackend, RasterExtrasOps, RasterPipeline, Result};

/// GPU work inserted before recording the native draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndirectCountPreprocess {
    None,
    CopyArguments,
    ComputeArguments,
}

/// How CPU recording grows with the declared maximum draw count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndirectCountRecording {
    Fixed,
    PerBatch,
    PerRecord,
}

/// Lowering of one nonempty multi-draw-count operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectCountRoute {
    /// The draw command consumes a GPU count buffer. The buffer can contain
    /// the caller's count or a count derived by preprocessing on the GPU.
    pub native_count: bool,
    pub preprocess: IndirectCountPreprocess,
    pub splits_raster_pass: bool,
    pub uses_temporary_resources: bool,
    pub recording: IndirectCountRecording,
}

/// Native batching for a nonempty GPU-counted operation with a fixed record layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectCountPlan {
    pub route: IndirectCountRoute,
    /// The number of native counted draw calls recorded for the declared
    /// maximum, regardless of the eventual value in the GPU count buffer.
    pub native_batches: u32,
}

/// If vertex/instance index adjustment is required by the selected pipeline,
/// select a route using the declared maximum count rather than the GPU value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectIndexAdjustment {
    pub small_batch_max: u32,
    pub small_batch: IndirectCountRoute,
    pub large_batch: IndirectCountRoute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectCountProfile {
    pub ordinary: IndirectCountRoute,
    pub adjusted_indices: Option<IndirectIndexAdjustment>,
}

impl IndirectCountProfile {
    /// A zero maximum emits no GPU work on routes publishing this profile.
    pub fn route(self, adjusts_indices: bool, maximum: u32) -> Option<IndirectCountRoute> {
        if maximum == 0 {
            return None;
        }
        Some(match self.adjusted_indices.filter(|_| adjusts_indices) {
            Some(adjustment) if maximum <= adjustment.small_batch_max => adjustment.small_batch,
            Some(adjustment) => adjustment.large_batch,
            None => self.ordinary,
        })
    }
}

impl<B: RasterBackend> Device<B> {
    /// Computes the selected route and native batch count before buffer
    /// allocation. Use `multi_draw_count_support` to qualify actual ranges.
    pub fn multi_draw_count_plan(
        &self,
        pipeline: &RasterPipeline<B>,
        maximum: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<Option<IndirectCountPlan>> {
        let Some(route) = self.multi_draw_count_route(pipeline, maximum)? else {
            return Ok(None);
        };
        let native_batches =
            B::Extras::multi_draw_count_batches(&pipeline.native, maximum, stride, indexed)?;
        Ok(Some(IndirectCountPlan {
            route,
            native_batches,
        }))
    }

    /// Checks argument and count buffer ranges and reports the selected route.
    /// Binding state and contents written by the GPU remain execution inputs.
    pub fn multi_draw_count_support(
        &self,
        pipeline: &RasterPipeline<B>,
        args: (&Buffer<B>, u64),
        count: (&Buffer<B>, u64),
        maximum: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<Option<IndirectCountRoute>> {
        if args.0.device != self.id || count.0.device != self.id {
            return Err(Error::WrongDevice);
        }
        let route = self.multi_draw_count_route(pipeline, maximum)?;
        B::Extras::multi_draw_count_support(
            (&args.0.native, args.1),
            (&count.0.native, count.1),
            maximum,
            stride,
            indexed,
        )?;
        Ok(route)
    }

    /// Looks up the lowering of an already-created pipeline without recording
    /// commands, allocating resources, or reading the GPU count.
    pub fn multi_draw_count_route(
        &self,
        pipeline: &RasterPipeline<B>,
        maximum: u32,
    ) -> Result<Option<IndirectCountRoute>> {
        if pipeline.device != self.id {
            return Err(Error::WrongDevice);
        }
        if !self.caps.multi_draw_count {
            return Err(Error::Unsupported("multi draw count"));
        }
        let profile = self
            .caps
            .multi_draw_count_profile
            .ok_or(Error::Unsupported("multi draw count lowering profile"))?;
        Ok(profile.route(
            B::adjusts_indirect_vertex_indices(&pipeline.native),
            maximum,
        ))
    }
}
