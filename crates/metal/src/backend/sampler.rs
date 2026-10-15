//! Metal sampler descriptor lowering.
//!
//! Only address modes and sampler fields with direct Metal equivalents are
//! accepted. Border sampling remains explicitly unsupported until platform
//! availability and border-color semantics can be verified on macOS devices.

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

        let native = MTLSamplerDescriptor::new();
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
        && desc.address.iter().all(|mode| {
            matches!(
                mode,
                AddressMode::ClampToEdge | AddressMode::Repeat | AddressMode::MirrorRepeat
            )
        })
}

#[cfg(target_vendor = "apple")]
fn address(mode: AddressMode) -> api::Result<objc2_metal::MTLSamplerAddressMode> {
    use objc2_metal::MTLSamplerAddressMode as Mtl;
    Ok(match mode {
        AddressMode::ClampToEdge => Mtl::ClampToEdge,
        AddressMode::Repeat => Mtl::Repeat,
        AddressMode::MirrorRepeat => Mtl::MirrorRepeat,
        AddressMode::ClampToBorder | AddressMode::ClampToZero => {
            return Err(api::Error::Unsupported("Metal sampler border address mode"));
        }
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
mod tests {
    use super::valid;
    use fluxel_rhi_core::api::{AddressMode, FilterMode, SamplerDesc};

    fn desc() -> SamplerDesc {
        SamplerDesc {
            label: None,
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..16.0,
            anisotropy: 16,
            compare: None,
            border: None,
        }
    }

    #[test]
    fn rejects_sampler_states_outside_the_native_subset() {
        assert!(valid(&desc()));
        let mut invalid = desc();
        invalid.address[0] = AddressMode::ClampToBorder;
        assert!(!valid(&invalid));
        let mut invalid = desc();
        invalid.anisotropy = 17;
        assert!(!valid(&invalid));
        let mut invalid = desc();
        invalid.lod = 2.0..1.0;
        assert!(!valid(&invalid));
    }
}
