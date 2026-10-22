//! Mesh dispatch and its command-list lifetime management.
use super::super::{Dx12, Dx12Buffer, Dx12Device, Dx12MeshPipeline, encoder::Dx12Encoder};
use fluxel_rhi_core::api::{self, Error, Result};
use std::future::{Ready, ready};
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

pub struct Mesh;
impl api::MeshOps<Dx12> for Mesh {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = Ready<Result<Dx12MeshPipeline>>;
    fn create_pipeline<'a>(
        device: &'a Dx12Device,
        desc: &'a api::MeshPipelineDesc<Dx12>,
    ) -> Self::PipelineFuture<'a> {
        ready(super::create_pipeline(device, desc))
    }
    fn bind_pipeline(encoder: &mut Dx12Encoder, pipeline: &Dx12MeshPipeline) -> Result<()> {
        if encoder.raster_format != Some(pipeline.0.color_format)
            || encoder.raster_samples != Some(pipeline.0.samples)
            || encoder.raster_multiview.is_some()
            || encoder.depth_target.is_some() != pipeline.0.depth_format.is_some()
        {
            return Err(Error::InvalidState);
        }
        // SAFETY: the encoder retains the PSO and root signature until its
        // submitted command list has retired.
        unsafe {
            encoder
                .list
                .SetGraphicsRootSignature(&pipeline.0.layout.root);
            encoder.list.SetPipelineState(&pipeline.0.state);
        }
        encoder.mesh_pipeline = Some(pipeline.clone());
        encoder.mesh_pipelines.push(pipeline.clone());
        encoder.pipeline = None;
        encoder.compute_pipeline = None;
        encoder.ray_pipeline = None;
        Ok(())
    }
    fn dispatch(encoder: &mut Dx12Encoder, groups: [u32; 3]) -> Result<()> {
        if encoder.mesh_pipeline.is_none() {
            return Err(Error::InvalidState);
        }
        if groups.contains(&0)
            || groups.iter().any(|&n| n > 65535)
            || groups
                .iter()
                .try_fold(1u64, |p, &n| p.checked_mul(u64::from(n)))
                .is_none_or(|n| n > (1 << 22))
        {
            return Err(Error::InvalidRange);
        }
        let list = encoder
            .list
            .cast::<ID3D12GraphicsCommandList6>()
            .map_err(super::super::native_error)?;
        // SAFETY: encoder owns the open list and retains the pipeline; counts
        // satisfy the D3D12 dispatch limits.
        unsafe { list.DispatchMesh(groups[0], groups[1], groups[2]) };
        Ok(())
    }
    fn dispatch_indirect(encoder: &mut Dx12Encoder, args: &Dx12Buffer, offset: u64) -> Result<()> {
        if encoder.mesh_pipeline.is_none() {
            return Err(Error::InvalidState);
        }
        if !args.usage.contains(api::BufferUsage::INDIRECT)
            || offset % 4 != 0
            || offset.checked_add(12).is_none_or(|end| end > args.size)
        {
            return Err(Error::InvalidRange);
        }
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0::default(),
        };
        let descriptor = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        let mut signature = None;
        // SAFETY: descriptor and argument live through synchronous creation;
        // no root signature is needed for this single dispatch operation.
        unsafe {
            encoder.device.0.native.CreateCommandSignature(
                &descriptor,
                None::<&ID3D12RootSignature>,
                &mut signature,
            )
        }
        .map_err(super::super::native_error)?;
        let signature: ID3D12CommandSignature = signature.ok_or(Error::OutOfMemory)?;
        // SAFETY: caller supplies the resource transition, and both signature
        // and buffer remain owned by the encoder until completion.
        unsafe {
            encoder.list.ExecuteIndirect(
                &signature,
                1,
                &args.resource,
                offset,
                None::<&ID3D12Resource>,
                0,
            )
        };
        encoder.mesh_signatures.push(signature);
        encoder.buffers.push(args.clone());
        Ok(())
    }
}
