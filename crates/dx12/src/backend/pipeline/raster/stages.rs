//! Native geometry and tessellation admission and PSO shader inputs.

use fluxel_rhi_core::api::{self, Error, Result, ShaderStage};
use windows::Win32::Graphics::Direct3D12::*;

use super::{
    AdmittedRaster, AdmittedStage, CompiledStage, Dx12, Dx12Device, Dx12ShaderSource,
    PipelineLayoutState, admission::admit_stage, bytecode, stage,
};
use crate::backend::shader_naga;

pub(super) struct AdditionalStages<T> {
    control: Option<T>,
    evaluation: Option<T>,
    geometry: Option<T>,
}

impl<'a> AdditionalStages<AdmittedStage<'a>> {
    pub(super) fn admit(
        device: &Dx12Device,
        desc: &'a api::RasterPipelineWithStagesDesc<Dx12>,
        base: &AdmittedRaster<'_>,
    ) -> Result<Self> {
        device.require_active()?;
        if device.0.shader_model < 60 {
            return Err(Error::Unsupported(
                "DX12 additional raster stages require SM 6.0",
            ));
        }
        if matches!(base.vertex.shader.source, Some(Dx12ShaderSource::Wgsl(_)))
            || base.fragment.as_ref().is_some_and(|fragment| {
                matches!(fragment.shader.source, Some(Dx12ShaderSource::Wgsl(_)))
            })
        {
            return Err(Error::Unsupported(
                "DX12 additional raster stages require a native shader chain",
            ));
        }
        let layout = base.layout;
        let geometry = desc
            .geometry
            .as_ref()
            .map(|geometry| admit_stage(geometry, ShaderStage::Geometry, layout))
            .transpose()?;
        let (control, evaluation) = match (desc.raster.primitive.topology, &desc.tessellation) {
            (api::PrimitiveTopology::PatchList(points), Some(tessellation)) => {
                if !(1..=D3D12_IA_PATCH_MAX_CONTROL_POINT_COUNT).contains(&points) {
                    return Err(Error::InvalidRange);
                }
                (
                    Some(admit_stage(
                        &tessellation.control,
                        ShaderStage::TessellationControl,
                        layout,
                    )?),
                    Some(admit_stage(
                        &tessellation.evaluation,
                        ShaderStage::TessellationEvaluation,
                        layout,
                    )?),
                )
            }
            (api::PrimitiveTopology::PatchList(_), None) | (_, Some(_)) => {
                return Err(Error::InvalidState);
            }
            _ if geometry.is_none() => return Err(Error::InvalidState),
            _ => (None, None),
        };
        if [control.as_ref(), evaluation.as_ref(), geometry.as_ref()]
            .into_iter()
            .flatten()
            .any(|stage| matches!(stage.shader.source, Some(Dx12ShaderSource::Wgsl(_))))
        {
            return Err(Error::Unsupported(
                "DX12 additional raster stages require native shaders",
            ));
        }
        Ok(Self {
            control,
            evaluation,
            geometry,
        })
    }

    pub(super) fn compile(
        &self,
        layout: &PipelineLayoutState,
        shader_model: u16,
        primitive: api::PrimitiveState,
    ) -> Result<AdditionalStages<CompiledStage>> {
        let compile = |input: &Option<AdmittedStage<'_>>| {
            input
                .as_ref()
                .map(|input| stage(input, layout, shader_model))
                .transpose()
        };
        let stages = AdditionalStages {
            control: compile(&self.control)?,
            evaluation: compile(&self.evaluation)?,
            geometry: compile(&self.geometry)?,
        };
        let [control, evaluation, geometry] = stages.bytecode();
        shader_naga::validate_classic_linkage(
            primitive.topology,
            control,
            evaluation,
            geometry,
            shader_model,
            primitive.conservative,
        )?;
        Ok(stages)
    }
}

impl AdditionalStages<CompiledStage> {
    pub(super) fn bytecode(&self) -> [&[u8]; 3] {
        [&self.control, &self.evaluation, &self.geometry].map(|stage| {
            stage
                .as_ref()
                .map_or(&[][..], |(bytes, ..)| bytes.as_slice())
        })
    }

    pub(super) fn set_bytecode(&self, pso: &mut D3D12_GRAPHICS_PIPELINE_STATE_DESC) {
        let [control, evaluation, geometry] = self.bytecode();
        for (target, bytes) in [
            (&mut pso.HS, control),
            (&mut pso.DS, evaluation),
            (&mut pso.GS, geometry),
        ] {
            if !bytes.is_empty() {
                *target = bytecode(bytes);
            }
        }
    }

    pub(super) fn diagnostics(self) -> impl Iterator<Item = api::PipelineShaderDiagnostics> {
        [self.control, self.evaluation, self.geometry]
            .into_iter()
            .flatten()
            .filter_map(|(_, _, _, report)| report)
    }
}
