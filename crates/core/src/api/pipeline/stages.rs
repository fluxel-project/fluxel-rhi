//! Optional classic raster stages and their patch input contract.

use super::*;

/// The tessellation control and evaluation stages are supplied as a pair.
#[derive(Clone)]
pub struct TessellationStages<B: Backend> {
    pub control: PipelineStage<B>,
    pub evaluation: PipelineStage<B>,
}

/// A classic raster pipeline with geometry, tessellation, or both.
///
/// The base primitive topology describes input assembly. With tessellation
/// it is a patch list. The native shader programs declare the generated
/// primitive shape and the linkage between consecutive stages.
#[derive(Clone)]
pub struct RasterPipelineWithStagesDesc<B: Backend> {
    pub raster: RasterPipelineDesc<B>,
    pub tessellation: Option<TessellationStages<B>>,
    pub geometry: Option<PipelineStage<B>>,
    pub dynamic_depth_bias: Option<DynamicDepthBias>,
    pub stencil_masks: Option<StencilMasks>,
}

impl<B: PipelineBackend> Device<B> {
    /// Qualifies the extended descriptor. Native signature linkage and PSO
    /// creation may still reject a combination of shader programs.
    pub fn raster_pipeline_with_stages_support(
        &self,
        desc: &RasterPipelineWithStagesDesc<B>,
    ) -> Result<()> {
        self.validate_raster_pipeline_stages(desc)?;
        B::raster_pipeline_with_stages_support(&self.native, desc)
    }

    pub async fn raster_pipeline_with_stages(
        &self,
        desc: RasterPipelineWithStagesDesc<B>,
    ) -> Result<RasterPipeline<B>> {
        self.validate_raster_pipeline_stages(&desc)?;
        let native = B::create_raster_pipeline_with_stages(&self.native, &desc).await?;
        // SAFETY: the base and additional shaders belong to this device.
        let mut pipeline = unsafe {
            RasterPipeline::from_native_with_dynamic_depth_bias(
                self.id,
                native,
                desc.dynamic_depth_bias,
            )
        };
        pipeline.layout = Some(desc.raster.layout);
        Ok(pipeline)
    }

    fn validate_raster_pipeline_stages(
        &self,
        desc: &RasterPipelineWithStagesDesc<B>,
    ) -> Result<()> {
        self.validate_raster_pipeline_desc(&desc.raster)?;
        if desc.geometry.is_none() && desc.tessellation.is_none() {
            return Err(Error::InvalidState);
        }
        if let Some(geometry) = &desc.geometry {
            if geometry.module.device != self.id {
                return Err(Error::WrongDevice);
            }
            if !self.caps.geometry_shader {
                return Err(Error::Unsupported("geometry shader"));
            }
        }
        match (desc.raster.primitive.topology, &desc.tessellation) {
            (PrimitiveTopology::PatchList(points), Some(tessellation)) => {
                if tessellation.control.module.device != self.id
                    || tessellation.evaluation.module.device != self.id
                {
                    return Err(Error::WrongDevice);
                }
                let maximum = self
                    .caps
                    .max_tessellation_patch_control_points
                    .ok_or(Error::Unsupported("tessellation shader"))?;
                if !(1..=maximum).contains(&points) {
                    return Err(Error::InvalidRange);
                }
                if desc.raster.primitive.strip_index_format.is_some() {
                    return Err(Error::InvalidState);
                }
            }
            (PrimitiveTopology::PatchList(_), None) | (_, Some(_)) => {
                return Err(Error::InvalidState);
            }
            _ => {}
        }
        if let Some(bias) = desc.dynamic_depth_bias {
            self.validate_dynamic_depth_bias(&desc.raster, bias)?;
        }
        if desc.stencil_masks.is_some() {
            self.validate_stencil_masks(desc.raster.depth_stencil)?;
        }
        Ok(())
    }
}
