//! Opaque D3D12 sampler-feedback resources and explicit clear/decode commands.

use super::{
    Dx12, Dx12Device, Dx12Texture,
    encoder::{self, Dx12Encoder},
    native_error,
    texture::TextureState,
};
use fluxel_rhi_core::api::{
    Error, Result, SamplerFeedbackBackend, SamplerFeedbackDesc, SamplerFeedbackMode,
    SamplerFeedbackState, SamplerFeedbackTier,
};
use std::sync::Arc;
use windows::{
    Win32::Graphics::{Direct3D12::*, Dxgi::Common::*},
    core::Interface,
};

pub(super) fn probe(device: &ID3D12Device, shader_model: u16) -> Option<SamplerFeedbackTier> {
    if shader_model < 65 || device.cast::<ID3D12Device8>().is_err() {
        return None;
    }
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    // SAFETY: the selector matches the initialized writable SDK structure.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS7,
            std::ptr::from_mut(&mut options).cast(),
            std::mem::size_of_val(&options) as u32,
        )
    }
    .ok()?;
    match options.SamplerFeedbackTier {
        D3D12_SAMPLER_FEEDBACK_TIER_1_0 => Some(SamplerFeedbackTier::Tier10),
        D3D12_SAMPLER_FEEDBACK_TIER_0_9 => Some(SamplerFeedbackTier::Tier09),
        _ => None,
    }
}

fn device8(device: &Dx12Device) -> Result<ID3D12Device8> {
    device.require_active()?;
    if device.0.sampler_feedback.is_none() {
        return Err(Error::Unsupported("DX12 sampler feedback"));
    }
    device
        .0
        .native
        .cast()
        .map_err(|_| Error::Unsupported("DX12 sampler feedback device interface"))
}

pub(super) fn write_uav(
    device: &Dx12Device,
    target: &Dx12Texture,
    map: &Dx12Texture,
    handle: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Result<()> {
    let native = device8(device)?;
    // SAFETY: the caller supplies a slot in the device's view heap and
    // provides the map and its paired target from the typed Core binding.
    unsafe {
        native.CreateSamplerFeedbackUnorderedAccessView(&target.0.resource, &map.0.resource, handle)
    };
    Ok(())
}

impl SamplerFeedbackBackend for Dx12 {
    fn create_feedback_map(
        device: &Dx12Device,
        target: &Dx12Texture,
        desc: SamplerFeedbackDesc,
    ) -> Result<Dx12Texture> {
        let native = device8(device)?;
        // SAFETY: the paired texture is retained and the query only reads its description.
        let paired = unsafe { target.0.resource.GetDesc() };
        if paired.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || paired.SampleDesc.Count != 1
            || paired
                .Flags
                .contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
        {
            return Err(Error::Unsupported("DX12 sampler feedback target"));
        }
        let [width, height] = desc.mip_region;
        let resource_desc = D3D12_RESOURCE_DESC1 {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: paired.Width,
            Height: paired.Height,
            DepthOrArraySize: paired.DepthOrArraySize,
            MipLevels: paired.MipLevels,
            Format: match desc.mode {
                SamplerFeedbackMode::MinMip => DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE,
                SamplerFeedbackMode::MipRegionUsed => {
                    DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE
                }
            },
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            SamplerFeedbackMipRegion: D3D12_MIP_REGION {
                Width: width,
                Height: height,
                Depth: 1,
            },
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };
        let mut resource = None;
        // SAFETY: both descriptors are initialized local records. DX12
        // writes the returned COM reference to the supplied output slot.
        unsafe {
            native.CreateCommittedResource2(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &resource_desc,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                None,
                None::<&ID3D12ProtectedResourceSession>,
                &mut resource,
            )
        }
        .map_err(native_error)?;
        Ok(Dx12Texture(Arc::new(TextureState {
            resource: resource.ok_or(Error::OutOfMemory)?,
            // Only the opaque-map domain owns this handle. Layout facts come
            // from the paired descriptor and ordinary texture APIs never see it.
            desc: target.0.desc.clone(),
            sparse: None,
            backing_heap: None,
            owns_residency: false,
            default_allocation: None,
        })))
    }

    fn clear_feedback_map(
        encoder: &mut Dx12Encoder,
        target: &Dx12Texture,
        map: &Dx12Texture,
    ) -> Result<()> {
        let native = device8(&encoder.device)?;
        let cpu = encoder
            .device
            .0
            .cpu_descriptors
            .allocate(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1)?;
        let gpu = encoder.device.0.descriptors.temporary_view()?;
        // SAFETY: the map is paired with the specified target. The command
        // list uses the device-wide shader-visible heap, and both descriptor
        // leases and native resources are retained below through completion.
        unsafe {
            native.CreateSamplerFeedbackUnorderedAccessView(
                &target.0.resource,
                &map.0.resource,
                cpu.cpu(0),
            );
            native.CopyDescriptorsSimple(
                1,
                gpu.cpu(),
                cpu.cpu(0),
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            );
            encoder.list.ClearUnorderedAccessViewUint(
                gpu.gpu(),
                cpu.cpu(0),
                &map.0.resource,
                &[0; 4],
                &[],
            );
        }
        encoder.view_descriptor_leases.push(cpu);
        encoder.dynamic_view_tables.push(Arc::new(gpu));
        encoder.textures.extend([target.clone(), map.clone()]);
        Ok(())
    }

    fn transition_feedback_map(
        encoder: &mut Dx12Encoder,
        map: &Dx12Texture,
        before: SamplerFeedbackState,
        after: SamplerFeedbackState,
    ) -> Result<()> {
        if encoder.device.0.sampler_feedback.is_none()
            || encoder.is_compute
                && (before == SamplerFeedbackState::ResolveSource
                    || after == SamplerFeedbackState::ResolveSource)
        {
            return Err(Error::Unsupported(
                "DX12 sampler feedback transition on this lane",
            ));
        }
        if before == SamplerFeedbackState::UnorderedAccess && after == before {
            encoder::uav_barrier(&encoder.list, &map.0.resource);
        } else {
            encoder::barrier(
                &encoder.list,
                &map.0.resource,
                resource_state(before),
                resource_state(after),
            );
        }
        encoder.textures.push(map.clone());
        Ok(())
    }

    fn decode_feedback_map(
        encoder: &mut Dx12Encoder,
        map: &Dx12Texture,
        destination: &Dx12Texture,
    ) -> Result<()> {
        if encoder.device.0.sampler_feedback.is_none() || encoder.is_compute {
            return Err(Error::Unsupported(
                "DX12 sampler feedback decode on this lane",
            ));
        }
        let list = encoder
            .list
            .cast::<ID3D12GraphicsCommandList1>()
            .map_err(native_error)?;
        // SAFETY: Core checks the matching destination shape and the caller
        // establishes both resolve states. UINT_MAX selects all subresources.
        unsafe {
            list.ResolveSubresourceRegion(
                &destination.0.resource,
                u32::MAX,
                0,
                0,
                &map.0.resource,
                u32::MAX,
                None,
                DXGI_FORMAT_R8_UINT,
                D3D12_RESOLVE_MODE_DECODE_SAMPLER_FEEDBACK,
            )
        };
        encoder.textures.extend([map.clone(), destination.clone()]);
        Ok(())
    }
}

fn resource_state(state: SamplerFeedbackState) -> D3D12_RESOURCE_STATES {
    match state {
        SamplerFeedbackState::UnorderedAccess => D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        SamplerFeedbackState::ResolveSource => D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
    }
}
