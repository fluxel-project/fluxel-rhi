//! Ray domain operations on a direct command list.
use super::super::{
    Dx12, Dx12Acceleration, Dx12BindGroup, Dx12Buffer, Dx12Device, Dx12RayPipeline,
    encoder::Dx12Encoder,
};
use fluxel_rhi_core::api::{self, Error, Result};
use std::{
    future::{Ready, ready},
    sync::Arc,
};
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

pub struct Ray;
fn same_layout(
    expected: &Arc<super::super::descriptor::BindGroupLayoutState>,
    actual: &Arc<super::super::descriptor::BindGroupLayoutState>,
) -> Result<()> {
    if Arc::ptr_eq(expected, actual) {
        Ok(())
    } else {
        Err(Error::InvalidState)
    }
}
impl api::RayOps<Dx12> for Ray {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = Ready<Result<Dx12RayPipeline>>;
    fn create_pipeline<'a>(
        device: &'a Dx12Device,
        desc: &'a api::RayPipelineDesc<Dx12>,
    ) -> Self::PipelineFuture<'a> {
        ready(super::pipeline::create_pipeline(device, desc))
    }
    fn set_pipeline(encoder: &mut Dx12Encoder, pipeline: &Dx12RayPipeline) -> Result<()> {
        let list = encoder
            .list
            .cast::<ID3D12GraphicsCommandList4>()
            .map_err(super::super::native_error)?;
        // SAFETY: the pipeline owns the state object and root signature and
        // the encoder retains it until its submission has completed.
        unsafe {
            list.SetComputeRootSignature(&pipeline.0.layout.root);
            list.SetPipelineState1(&pipeline.0.state);
        }
        encoder.ray_pipeline = Some(pipeline.clone());
        encoder.ray_pipelines.push(pipeline.clone());
        encoder.compute_pipeline = None;
        encoder.pipeline = None;
        encoder.mesh_pipeline = None;
        Ok(())
    }
    fn set_bind_group(
        encoder: &mut Dx12Encoder,
        slot: u32,
        group: &Dx12BindGroup,
        offsets: &[u32],
    ) -> Result<()> {
        if !offsets.is_empty() {
            return Err(Error::Unsupported("DX12 ray dynamic offsets"));
        }
        let pipeline = encoder.ray_pipeline.as_ref().ok_or(Error::InvalidState)?;
        let expected = pipeline
            .0
            .layout
            .groups
            .get(slot as usize)
            .ok_or(Error::InvalidRange)?;
        same_layout(expected, &group.0.layout)?;
        let table = pipeline
            .0
            .layout
            .tables
            .get(slot as usize)
            .ok_or(Error::InvalidRange)?;
        if !table.dynamic_cbvs.is_empty()
            || group.0.has_views() != table.views.is_some()
            || group.0.has_samplers() != table.samplers.is_some()
        {
            return Err(Error::InvalidState);
        }
        // SAFETY: heaps and descriptors belong to the retained bind group.
        unsafe {
            let heaps = [
                Some(group.0.heaps().view_heap().clone()),
                Some(group.0.heaps().sampler_heap().clone()),
            ];
            encoder.list.SetDescriptorHeaps(&heaps);
            if let Some(root) = table.views {
                encoder
                    .list
                    .SetComputeRootDescriptorTable(root, group.0.view_gpu());
            }
            if let Some(root) = table.samplers {
                encoder
                    .list
                    .SetComputeRootDescriptorTable(root, group.0.sampler_gpu());
            }
        }
        encoder.bound_groups.push(group.clone());
        Ok(())
    }
    fn shader_table_properties(device: &Dx12Device) -> Result<api::ShaderTableProperties> {
        super::ray_device(device)?;
        Ok(api::ShaderTableProperties {
            identifier_size: D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES,
            record_alignment: D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT,
            table_alignment: D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT,
            max_record_stride: D3D12_RAYTRACING_MAX_SHADER_RECORD_STRIDE,
        })
    }
    fn shader_group_identifier(
        device: &Dx12Device,
        pipeline: &Dx12RayPipeline,
        index: u32,
    ) -> Result<Vec<u8>> {
        super::ray_device(device)?;
        super::pipeline::identifier(pipeline, index)
    }
    fn acceleration_sizes(
        device: &Dx12Device,
        desc: &api::AccelerationDesc<Dx12>,
    ) -> Result<api::AccelerationSizes> {
        super::sizes(device, desc)
    }
    fn create_acceleration(
        device: &Dx12Device,
        desc: &api::AccelerationDesc<Dx12>,
    ) -> Result<Dx12Acceleration> {
        super::create_acceleration(device, desc)
    }
    fn build_acceleration(
        encoder: &mut Dx12Encoder,
        dst: &Dx12Acceleration,
        scratch: &Dx12Buffer,
        desc: &api::AccelerationDesc<Dx12>,
    ) -> Result<()> {
        super::build(encoder, dst, scratch, desc)
    }
    fn trace(
        encoder: &mut Dx12Encoder,
        pipeline: &Dx12RayPipeline,
        table: &api::ShaderTable<Dx12>,
        dims: [u32; 3],
    ) -> Result<()> {
        if !encoder
            .ray_pipeline
            .as_ref()
            .is_some_and(|bound| Arc::ptr_eq(&bound.0, &pipeline.0))
        {
            return Err(Error::InvalidState);
        }
        super::trace::dispatch(encoder, table, dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ray_binding_requires_pipeline_group_layout_identity() {
        let first =
            Arc::new(super::super::super::descriptor::BindGroupLayoutState { slots: vec![] });
        let different =
            Arc::new(super::super::super::descriptor::BindGroupLayoutState { slots: vec![] });
        assert!(same_layout(&first, &first.clone()).is_ok());
        assert!(same_layout(&first, &different).is_err());
    }
}
