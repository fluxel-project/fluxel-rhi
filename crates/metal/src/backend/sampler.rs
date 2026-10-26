//! Metal sampler descriptor lowering.
//!
//! Border color is gated on the Metal GPU family; clamp-to-zero is native.

use fluxel_rhi_core::api::{self, AddressMode, SamplerDesc};
#[cfg(target_vendor = "apple")]
use fluxel_rhi_core::api::{CompareFunction, FilterMode};

use super::MetalDevice;

#[derive(Clone)]
pub struct MetalSampler {
    #[cfg(target_vendor = "apple")]
    #[allow(dead_code)]
    pub(crate) raw:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSamplerState>>,
    #[cfg(target_vendor = "apple")]
    pub(crate) filtering: bool,
    #[cfg(target_vendor = "apple")]
    pub(crate) comparison: bool,
}

pub(super) fn create_sampler(
    device: &MetalDevice,
    desc: &SamplerDesc,
) -> api::Result<MetalSampler> {
    if !valid(desc) {
        return Err(api::Error::Unsupported("Metal sampler descriptor"));
    }
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::NSString;
        use objc2_metal::{MTLDevice as _, MTLSamplerDescriptor, MTLSamplerMipFilter};

        if desc.address.contains(&AddressMode::ClampToBorder) {
            use objc2_metal::MTLGPUFamily;
            if !device._state._device.supportsFamily(MTLGPUFamily::Apple7)
                && !device._state._device.supportsFamily(MTLGPUFamily::Mac2)
            {
                return Err(api::Error::Unsupported("Metal border color GPU family"));
            }
        }

        let native = MTLSamplerDescriptor::new();
        if let Some(color) = desc.border {
            use api::BorderColor;
            use objc2_metal::MTLSamplerBorderColor as Mtl;
            native.setBorderColor(match color {
                BorderColor::TransparentBlack | BorderColor::ZeroInteger => Mtl::TransparentBlack,
                BorderColor::OpaqueBlack => Mtl::OpaqueBlack,
                BorderColor::OpaqueWhite => Mtl::OpaqueWhite,
            });
        }
        native.setSAddressMode(address(desc.address[0])?);
        native.setTAddressMode(address(desc.address[1])?);
        native.setRAddressMode(address(desc.address[2])?);
        native.setMagFilter(filter(desc.mag_filter));
        native.setMinFilter(filter(desc.min_filter));
        native.setMipFilter(match desc.mip_filter {
            FilterMode::Nearest => MTLSamplerMipFilter::Nearest,
            FilterMode::Linear => MTLSamplerMipFilter::Linear,
        });
        native.setLodMinClamp(desc.lod.start);
        native.setLodMaxClamp(desc.lod.end);
        native.setMaxAnisotropy(desc.anisotropy as usize);
        if let Some(compare) = desc.compare {
            native.setCompareFunction(compare_function(compare));
        }
        if let Some(label) = desc.label.as_deref() {
            native.setLabel(Some(&NSString::from_str(label)));
        }
        let raw = device
            ._state
            ._device
            .newSamplerStateWithDescriptor(&native)
            .ok_or(api::Error::OutOfMemory)?;
        Ok(MetalSampler {
            raw,
            filtering: desc.mag_filter == FilterMode::Linear
                || desc.min_filter == FilterMode::Linear
                || desc.mip_filter == FilterMode::Linear,
            comparison: desc.compare.is_some(),
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = device;
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

fn valid(desc: &SamplerDesc) -> bool {
    desc.anisotropy > 0
        && desc.anisotropy <= 16
        && desc.lod.start.is_finite()
        && desc.lod.end.is_finite()
        && desc.lod.start >= 0.0
        && desc.lod.start <= desc.lod.end
        && (!desc.address.contains(&AddressMode::ClampToBorder) || desc.border.is_some())
}

#[cfg(target_vendor = "apple")]
fn address(mode: AddressMode) -> api::Result<objc2_metal::MTLSamplerAddressMode> {
    use objc2_metal::MTLSamplerAddressMode as Mtl;
    Ok(match mode {
        AddressMode::ClampToEdge => Mtl::ClampToEdge,
        AddressMode::Repeat => Mtl::Repeat,
        AddressMode::MirrorRepeat => Mtl::MirrorRepeat,
        AddressMode::ClampToBorder => Mtl::ClampToBorderColor,
        AddressMode::ClampToZero => Mtl::ClampToZero,
    })
}

#[cfg(target_vendor = "apple")]
fn filter(mode: FilterMode) -> objc2_metal::MTLSamplerMinMagFilter {
    use objc2_metal::MTLSamplerMinMagFilter as Mtl;
    match mode {
        FilterMode::Nearest => Mtl::Nearest,
        FilterMode::Linear => Mtl::Linear,
    }
}

#[cfg(target_vendor = "apple")]
fn compare_function(value: CompareFunction) -> objc2_metal::MTLCompareFunction {
    use objc2_metal::MTLCompareFunction as Mtl;
    match value {
        CompareFunction::Never => Mtl::Never,
        CompareFunction::Less => Mtl::Less,
        CompareFunction::Equal => Mtl::Equal,
        CompareFunction::LessEqual => Mtl::LessEqual,
        CompareFunction::Greater => Mtl::Greater,
        CompareFunction::NotEqual => Mtl::NotEqual,
        CompareFunction::GreaterEqual => Mtl::GreaterEqual,
        CompareFunction::Always => Mtl::Always,
    }
}

#[cfg(test)]
#[path = "../tests/backend/sampler.rs"]
mod tests;
