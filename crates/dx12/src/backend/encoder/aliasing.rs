//! Explicit alias activation barriers for caller-placed resources.
//!
//! Native barriers preserve the caller's ordering and nullable before/after
//! resource selection. They do not infer heap overlap or resource lifetimes.

use super::*;

fn retain(
    encoder: &mut Dx12Encoder,
    resource: &Option<api::AliasedResource<'_, Dx12>>,
) -> Option<ID3D12Resource> {
    match resource.as_ref()? {
        api::AliasedResource::Buffer(buffer) => {
            // SAFETY: Core checked this barrier's device identity. Retaining
            // its native owner also keeps the placed heap alive through GPU use.
            let buffer = unsafe { buffer.native_ref() }.clone();
            let native = buffer.resource.clone();
            encoder.buffers.push(buffer);
            Some(native)
        }
        api::AliasedResource::Texture(texture) => {
            // SAFETY: same checked device and retained-resource invariant.
            let texture = unsafe { texture.native_ref() }.clone();
            let native = texture.0.resource.clone();
            encoder.textures.push(texture);
            Some(native)
        }
        api::AliasedResource::Acceleration(acceleration) => {
            // SAFETY: same checked device and retained-resource invariant.
            let acceleration = unsafe { acceleration.native_ref() }.clone();
            let native = acceleration.0.resource.clone();
            encoder.accelerations.push(acceleration);
            Some(native)
        }
    }
}

pub(super) fn record(
    encoder: &mut Dx12Encoder,
    before: &Option<api::AliasedResource<'_, Dx12>>,
    after: &Option<api::AliasedResource<'_, Dx12>>,
) {
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                pResourceBefore: ManuallyDrop::new(retain(encoder, before)),
                pResourceAfter: ManuallyDrop::new(retain(encoder, after)),
            }),
        },
    };
    // SAFETY: nullable COM resources live through this synchronous recording
    // call; their typed owners remain in encoder retention until completion.
    // Release exactly the temporary clones inside the FFI union afterwards.
    unsafe {
        encoder.list.ResourceBarrier(std::slice::from_ref(&barrier));
        ManuallyDrop::drop(&mut (*barrier.Anonymous.Aliasing).pResourceBefore);
        ManuallyDrop::drop(&mut (*barrier.Anonymous.Aliasing).pResourceAfter);
    }
}
