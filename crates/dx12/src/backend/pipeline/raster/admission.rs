//! DX12 raster descriptor checks shared by support queries and PSO creation.

use std::sync::Arc;

use fluxel_rhi_core::api::{self, Error, Result};
use windows::{Win32::Graphics::Direct3D12::*, core::PCSTR};

use crate::backend::{Dx12, Dx12Device, Dx12Shader, Dx12ShaderSource};

use super::super::fixed::{Attachments, lower_attachments, lower_raster};
use super::{PipelineLayoutState, admitted_vertex_format};

pub(super) struct AdmittedRaster<'a> {
    pub(super) layout: &'a Arc<PipelineLayoutState>,
    pub(super) vertex: AdmittedStage<'a>,
    pub(super) fragment: Option<AdmittedStage<'a>>,
    pub(super) attachments: Attachments,
    pub(super) vertex_strides: Vec<u32>,
    pub(super) elements: Vec<D3D12_INPUT_ELEMENT_DESC>,
    pub(super) raster: D3D12_RASTERIZER_DESC,
}

pub(super) struct AdmittedStage<'a> {
    pub(super) desc: &'a api::PipelineStage<Dx12>,
    pub(super) shader: &'a Dx12Shader,
    pub(super) entry: &'a api::ShaderEntryPoint,
}

pub(super) fn admit<'a>(
    device: &Dx12Device,
    desc: &'a api::RasterPipelineDesc<Dx12>,
) -> Result<AdmittedRaster<'a>> {
    if desc.primitive.strip_index_format.is_some()
        && !matches!(
            desc.primitive.topology,
            api::PrimitiveTopology::LineStrip
                | api::PrimitiveTopology::TriangleStrip
                | api::PrimitiveTopology::LineStripAdjacency
                | api::PrimitiveTopology::TriangleStripAdjacency
        )
    {
        return Err(Error::InvalidState);
    }
    if desc
        .multiview
        .is_some_and(|views| !(1..=D3D12_MAX_VIEW_INSTANCE_COUNT).contains(&views))
    {
        return Err(Error::Unsupported("DX12 multiview view count"));
    }
    let attachments = lower_attachments(
        device,
        &desc.color_targets,
        desc.depth_stencil,
        desc.multisample,
    )?;
    // SAFETY: Core checked layout provenance and the descriptor retains its native state.
    let layout = &unsafe { desc.layout.native_ref() }.0;
    let vertex = admit_stage(&desc.vertex, api::ShaderStage::Vertex, layout)?;
    let fragment = desc
        .fragment
        .as_ref()
        .map(|stage| admit_stage(stage, api::ShaderStage::Fragment, layout))
        .transpose()?;
    // The command lowering needs the public strides to construct D3D12 vertex
    // buffer views.  Keep this validation adjacent to input-layout lowering so
    // a pipeline can never be created with a stride the command path cannot
    // represent.
    let vertex_strides: Result<Vec<u32>> = desc
        .vertex_buffers
        .iter()
        .map(|buffer| u32::try_from(buffer.stride).map_err(|_| Error::InvalidRange))
        .collect();
    let vertex_strides = vertex_strides?;
    // Naga emits LOC<n>; native artifacts follow the LOCATION<n> ABI.
    // Both names have static storage through the native PSO call.
    let semantic_name = if matches!(vertex.shader.source, Some(Dx12ShaderSource::Wgsl(_))) {
        PCSTR(c"LOC".as_ptr().cast())
    } else {
        PCSTR(c"LOCATION".as_ptr().cast())
    };
    let elements: Result<Vec<_>> = desc
        .vertex_buffers
        .iter()
        .enumerate()
        .flat_map(|(slot, b)| b.attributes.iter().map(move |a| (slot, b, a)))
        .map(|(slot, b, a)| {
            let offset = u32::try_from(a.offset).map_err(|_| Error::InvalidRange)?;
            // Core offsets are literal bytes. DX12 reserves this bit pattern
            // to compute the offset from the preceding input element.
            if offset == D3D12_APPEND_ALIGNED_ELEMENT {
                return Err(Error::InvalidRange);
            }
            Ok(D3D12_INPUT_ELEMENT_DESC {
                SemanticName: semantic_name,
                SemanticIndex: a.location,
                Format: admitted_vertex_format(device, a.format)?,
                InputSlot: slot as u32,
                AlignedByteOffset: offset,
                InputSlotClass: match b.step_mode {
                    api::VertexStepMode::Vertex => D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    api::VertexStepMode::Instance => D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                },
                InstanceDataStepRate: match b.step_mode {
                    api::VertexStepMode::Vertex => 0,
                    api::VertexStepMode::Instance => 1,
                },
            })
        })
        .collect();
    let elements = elements?;
    let raster = lower_raster(desc.primitive, desc.multisample, attachments.depth)?;
    Ok(AdmittedRaster {
        layout,
        vertex,
        fragment,
        attachments,
        vertex_strides,
        elements,
        raster,
    })
}

pub(super) fn admit_stage<'a>(
    desc: &'a api::PipelineStage<Dx12>,
    need: api::ShaderStage,
    layout: &PipelineLayoutState,
) -> Result<AdmittedStage<'a>> {
    // SAFETY: Core checked shader provenance and the descriptor retains the module.
    let shader = unsafe { desc.module.native_ref() };
    let entry = shader
        .entries
        .iter()
        .find(|e| e.name == desc.entry && e.stage == need)
        .ok_or_else(|| {
            Error::Shader(format!(
                "DXIL has no requested {:?} entry {}",
                need, desc.entry
            ))
        })?;
    if !layout.native_abi_compatible {
        match shader.source {
            Some(Dx12ShaderSource::Hlsl(_)) => {
                return Err(Error::Unsupported("DX12 HLSL descriptor register overlap"));
            }
            None => return Err(Error::Unsupported("DX12 DXIL descriptor register overlap")),
            Some(Dx12ShaderSource::Wgsl(_)) => {}
        }
    }
    if shader.source.is_none() {
        if !desc.constants.is_empty() {
            return Err(Error::Unsupported("DX12 DXIL specialization"));
        }
        crate::backend::shader_naga::validate_dxil_visibility(&shader.bytes, need, layout)?;
    }
    Ok(AdmittedStage {
        desc,
        shader,
        entry,
    })
}
