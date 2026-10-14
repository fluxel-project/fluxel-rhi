//! Core resource, view, sampler, and binding-layout lowering.
//!
//! This module owns immutable resource descriptions and direct native object
//! allocation. Bind groups retain resources for lifetime only; applying them
//! remains an immediate raster verb.

use fluxel_rhi_core::api::{
    self, BindingResource, BufferDesc, BufferSupport, Error, ResourceBackend, Result, Support,
    TextureDesc, TextureSupport,
};

mod binding;
pub(crate) use binding::lower_program_bindings;

use super::{Gl, GlDevice, spi::native_error, types::*};

impl ResourceBackend for Gl {
    fn buffer_support(_: &GlDevice, desc: &BufferDesc) -> Support<BufferSupport> {
        match crate::backend::gl::translate::lower_buffer_desc(desc) {
            Ok(_) => Support::Supported(BufferSupport {
                alignment: 1,
                max_size: i32::MAX as u64,
            }),
            Err(Error::Unsupported(reason)) => Support::Unsupported(reason),
            Err(_) => Support::Unsupported("invalid GL buffer description"),
        }
    }

    fn texture_support(device: &GlDevice, desc: &TextureDesc) -> Support<TextureSupport> {
        match crate::backend::gl::translate::lower_texture_desc(desc) {
            Ok(lowered) => texture_support(device, lowered),
            Err(Error::Unsupported(reason)) => Support::Unsupported(reason),
            Err(_) => Support::Unsupported("invalid GL texture description"),
        }
    }

    fn create_buffer(device: &GlDevice, desc: &BufferDesc) -> Result<GlBuffer> {
        let desc = crate::backend::gl::translate::lower_buffer_desc(desc)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::GlResourceApi as _;

            let id = device
                .0
                .with_native("create Core buffer", |provider| {
                    provider.create_buffer_resource(desc)
                })
                .map_err(native_error)?;
            return Ok(GlBuffer::from_inner(GlBufferInner {
                state: device.0.clone(),
                id,
                desc,
            }));
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            {
                let browser = device.0.owner.create_buffer(desc)?;
                return Ok(GlBuffer::from_inner(GlBufferInner {
                    state: device.0.clone(),
                    desc,
                    browser,
                }));
            }
            #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
            {
                let _ = (device, desc);
                Err(Error::Unsupported("GL platform context"))
            }
        }
    }

    fn create_texture(device: &GlDevice, desc: &TextureDesc) -> Result<GlTexture> {
        let desc = crate::backend::gl::translate::lower_texture_desc(desc)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::GlResourceApi as _;

            let id = device
                .0
                .with_native("create Core texture", |provider| {
                    provider.create_texture_resource(desc)
                })
                .map_err(native_error)?;
            return Ok(GlTexture::from_inner(GlTextureInner {
                state: device.0.clone(),
                id,
                desc,
            }));
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            {
                let browser = device.0.owner.create_texture(desc)?;
                return Ok(GlTexture::from_inner(GlTextureInner {
                    state: device.0.clone(),
                    desc,
                    browser,
                }));
            }
            #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
            {
                let _ = (device, desc);
                Err(Error::Unsupported("GL platform context"))
            }
        }
    }

    fn create_view(
        _: &GlDevice,
        texture: &GlTexture,
        desc: &api::TextureViewDesc,
    ) -> Result<GlTextureView> {
        let allocated = texture.desc;
        let (layers, aspects) = match allocated.dimension {
            crate::backend::gl::api::GlTextureDimension::D2 => (
                1,
                if allocated.format == crate::backend::gl::api::GlFormat::Depth32Float {
                    api::TextureAspects::DEPTH
                } else {
                    api::TextureAspects::COLOR
                },
            ),
            crate::backend::gl::api::GlTextureDimension::D2Array => {
                (allocated.extent.depth_or_layers, api::TextureAspects::COLOR)
            }
            _ => return Err(Error::Unsupported("GL texture view dimension")),
        };
        let dimension_matches = matches!(
            (allocated.dimension, desc.dimension),
            (
                crate::backend::gl::api::GlTextureDimension::D2,
                api::TextureViewDimension::D2
            ) | (
                crate::backend::gl::api::GlTextureDimension::D2Array,
                api::TextureViewDimension::D2Array
            )
        );
        if !dimension_matches || desc.aspects != aspects {
            return Err(Error::Unsupported("GL texture view dimension or aspect"));
        }
        if crate::backend::gl::translate::lower_texture_format(desc.format)? != allocated.format {
            return Err(Error::Unsupported(
                "GL texture view format reinterpretation",
            ));
        }
        // A sampled GL texture exposes the allocation's mip range and array
        // layers as one binding. Narrow subranges would require a separate GL
        // texture-view object; do not silently sample outside the Core view.
        let full_sampling_view =
            desc.mips == (0..allocated.mip_level_count) && desc.layers == (0..layers);
        let single_attachment_view = desc.mips == (0..1)
            && desc.layers == (0..1)
            && allocated.dimension == crate::backend::gl::api::GlTextureDimension::D2;
        if !full_sampling_view && !single_attachment_view {
            return Err(Error::Unsupported("GL texture view subrange"));
        }
        let extent = allocated
            .mip_extent(desc.mips.start)
            .ok_or(Error::InvalidRange)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            return Ok(GlTextureView {
                texture: texture.clone(),
                native: crate::backend::gl::api::GlTextureView {
                    target: crate::backend::gl::api::GlAttachmentTarget::Texture(texture.id),
                    format: allocated.format,
                    mip_level: desc.mips.start,
                    array_layer: desc.layers.start,
                    layer_count: desc.layers.end - desc.layers.start,
                    width: extent.width,
                    height: extent.height,
                    sample_count: allocated.sample_count,
                },
            });
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        Ok(GlTextureView {
            texture: texture.clone(),
        })
    }

    fn create_sampler(device: &GlDevice, desc: &api::SamplerDesc) -> Result<GlSampler> {
        if desc.border.is_some() {
            return Err(Error::Unsupported("GL sampler border color"));
        }
        let native = crate::backend::gl::api::GlSamplerDesc {
            address_mode_u: lower_address(desc.address[0])?,
            address_mode_v: lower_address(desc.address[1])?,
            address_mode_w: lower_address(desc.address[2])?,
            mag_filter: lower_filter(desc.mag_filter),
            min_filter: lower_filter(desc.min_filter),
            mipmap_filter: lower_mip_filter(desc.mip_filter),
            lod_min_bits: desc.lod.start.to_bits(),
            lod_max_bits: desc.lod.end.to_bits(),
            compare: desc.compare.map(lower_compare),
            max_anisotropy_bits: (desc.anisotropy > 1)
                .then(|| f32::from(desc.anisotropy).to_bits()),
        };
        // The native provider and browser owner both validate optional
        // anisotropy against extension and limit evidence captured for this
        // exact context before mutating GL.
        native.validate().map_err(|_| Error::InvalidRange)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::GlSamplerApi as _;
            let id = device
                .0
                .with_native("create Core GL sampler", |provider| {
                    provider.create_sampler(native)
                })
                .map_err(native_error)?;
            return Ok(GlSampler::from_inner(GlSamplerInner {
                state: device.0.clone(),
                id,
            }));
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            {
                native
                    .validate_for(&device.0.owner.discovery_snapshot())
                    .map_err(|error| match error {
                        crate::backend::gl::api::GlSamplerValidationError::AnisotropyExtensionUnavailable
                        | crate::backend::gl::api::GlSamplerValidationError::AnisotropyLimitUnavailable
                        | crate::backend::gl::api::GlSamplerValidationError::AnisotropyExceedsLimit => {
                            Error::Unsupported("WebGL2 sampler anisotropy is unavailable")
                        }
                        _ => Error::InvalidRange,
                    })?;
                let browser = device.0.owner.create_sampler(native)?;
                return Ok(GlSampler::from_inner(GlSamplerInner {
                    state: device.0.clone(),
                    browser,
                }));
            }
            #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
            {
                let _ = (device, native);
                Err(Error::Unsupported("GL platform context"))
            }
        }
    }

    fn create_bind_group_layout(
        device: &GlDevice,
        desc: &api::BindGroupLayoutDesc,
    ) -> Result<GlBindGroupLayout> {
        let entries = binding::lower_bind_group_layout(desc)?;
        Ok(GlBindGroupLayout {
            state: device.0.clone(),
            entries: entries.into(),
        })
    }

    fn create_bind_group(
        device: &GlDevice,
        desc: &api::BindGroupDesc<'_, Gl>,
    ) -> Result<GlBindGroup> {
        // SAFETY: Core holds the layout borrow for this synchronous creation
        // call. The backend reads immutable entries and clones retained native
        // resource wrappers into the returned group.
        let layout = unsafe { desc.layout.native_ref() };
        if !std::rc::Rc::ptr_eq(&device.0, &layout.state) {
            return Err(Error::InvalidState);
        }
        if desc.entries.len() != layout.entries.len() {
            return Err(Error::InvalidState);
        }
        let mut resources = Vec::with_capacity(desc.entries.len());
        for expected in layout.entries.iter() {
            let entry = desc
                .entries
                .iter()
                .find(|entry| entry.binding == expected.binding)
                .ok_or(Error::InvalidState)?;
            let resource = match (expected.slot, &entry.resource) {
                (
                    GlBindingSlot::UniformBuffer { min_size, .. },
                    BindingResource::Buffer { buffer, range },
                ) => {
                    // SAFETY: Core validates resource provenance before this
                    // SPI call. The wrapper is borrowed for this synchronous
                    // extraction, then cloned into the returned bind group.
                    let buffer = unsafe { buffer.native_ref() };
                    let offset = u32::try_from(range.start).map_err(|_| Error::InvalidRange)?;
                    let size = range
                        .end
                        .checked_sub(range.start)
                        .ok_or(Error::InvalidRange)?;
                    let size = u32::try_from(size).map_err(|_| Error::InvalidRange)?;
                    if range.end > buffer.desc.size || size == 0 || u64::from(size) < min_size {
                        return Err(Error::InvalidRange);
                    }
                    if !std::rc::Rc::ptr_eq(&device.0, &buffer.state) {
                        return Err(Error::InvalidState);
                    }
                    GlBoundResource::UniformBuffer {
                        buffer: buffer.clone(),
                        offset,
                        size,
                    }
                }
                (
                    GlBindingSlot::SampledTexture { dimension },
                    BindingResource::TextureView(view),
                ) => {
                    // SAFETY: see the buffer arm; views retain their textures.
                    let view = unsafe { view.native_ref() };
                    let view_dimension = match view.texture.desc.dimension {
                        crate::backend::gl::api::GlTextureDimension::D2 => {
                            api::TextureViewDimension::D2
                        }
                        crate::backend::gl::api::GlTextureDimension::D2Array => {
                            api::TextureViewDimension::D2Array
                        }
                        _ => return Err(Error::Unsupported("GL sampled texture dimension")),
                    };
                    if !std::rc::Rc::ptr_eq(&device.0, &view.texture.state)
                        || dimension != view_dimension
                    {
                        return Err(Error::InvalidState);
                    }
                    GlBoundResource::Texture(view.clone())
                }
                (
                    GlBindingSlot::StorageTexture {
                        dimension, format, ..
                    },
                    BindingResource::TextureView(view),
                ) => {
                    // Storage images are one native image view.  The current
                    // Core route deliberately admits only a complete 2D view:
                    // layered views require a separate lowerer and must not be
                    // silently rebound as layer zero.
                    let view = unsafe { view.native_ref() };
                    let view_dimension = match view.texture.desc.dimension {
                        crate::backend::gl::api::GlTextureDimension::D2 => {
                            api::TextureViewDimension::D2
                        }
                        crate::backend::gl::api::GlTextureDimension::D2Array => {
                            api::TextureViewDimension::D2Array
                        }
                        _ => return Err(Error::Unsupported("GL storage texture dimension")),
                    };
                    if !std::rc::Rc::ptr_eq(&device.0, &view.texture.state)
                        || dimension != view_dimension
                        || crate::backend::gl::translate::lower_texture_format(format)?
                            != view.texture.desc.format
                    {
                        return Err(Error::InvalidState);
                    }
                    GlBoundResource::StorageTexture(view.clone())
                }
                (GlBindingSlot::Sampler, BindingResource::Sampler(sampler)) => {
                    // SAFETY: see the buffer arm; sampler clones share one
                    // provider allocation through their Rc inner object.
                    let sampler = unsafe { sampler.native_ref() };
                    if !std::rc::Rc::ptr_eq(&device.0, &sampler.state) {
                        return Err(Error::InvalidState);
                    }
                    GlBoundResource::Sampler(sampler.clone())
                }
                _ => return Err(Error::InvalidState),
            };
            resources.push((expected.binding, resource));
        }
        Ok(GlBindGroup::from_inner(GlBindGroupInner {
            layout: layout.clone(),
            resources: resources.into(),
        }))
    }

    fn create_pipeline_layout(
        device: &GlDevice,
        desc: &api::PipelineLayoutDesc<'_, Gl>,
    ) -> Result<GlPipelineLayout> {
        if !desc.push_constants.is_empty() {
            return Err(Error::Unsupported("GL pipeline layout shape"));
        }
        let mut groups = Vec::with_capacity(desc.groups.len());
        let mut bindings = Vec::new();
        for (index, group) in desc.groups.iter().enumerate() {
            // SAFETY: Core retains each layout borrow throughout synchronous
            // pipeline-layout creation. Only immutable ABI entries are read.
            let group = unsafe { group.native_ref() };
            if !std::rc::Rc::ptr_eq(&device.0, &group.state) {
                return Err(Error::InvalidState);
            }
            let index = u32::try_from(index).map_err(|_| Error::InvalidRange)?;
            bindings.extend(lower_program_bindings(index, &group.entries)?);
            groups.push(group.clone());
        }
        Ok(GlPipelineLayout {
            state: device.0.clone(),
            groups: groups.into(),
            native: crate::backend::gl::api::GlPipelineLayout { bindings },
        })
    }
}

/// Reports only format/count/size combinations proven by this context's
/// discovery snapshot. `create_texture` performs the final native allocation;
/// no guessed GL specification maximum is published through Core.
fn texture_support(
    device: &GlDevice,
    desc: crate::backend::gl::api::GlTextureDesc,
) -> Support<TextureSupport> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        return device
            .0
            .with_native("query GL texture support", |provider| {
                Ok(texture_support_from_discovery(
                    provider.discovery(),
                    desc,
                    false,
                ))
            })
            .unwrap_or(Support::Unsupported("GL context unavailable"));
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            let facts = device.0.owner.discovery_snapshot();
            return texture_support_from_discovery(&facts, desc, true);
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        {
            let _ = (device, desc);
            Support::Unsupported("GL platform context")
        }
    }
}

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    feature = "native-gles-egl",
    all(target_arch = "wasm32", feature = "webgl2")
))]
fn texture_support_from_discovery(
    discovery: &crate::backend::gl::api::GlDiscoverySnapshot,
    desc: crate::backend::gl::api::GlTextureDesc,
    browser: bool,
) -> Support<TextureSupport> {
    use crate::backend::gl::api::{GlTextureDimension, GlTextureUsage};
    if !matches!(
        desc.dimension,
        GlTextureDimension::D2 | GlTextureDimension::D2Array
    ) {
        return Support::Unsupported("GL texture dimension");
    }
    if browser && desc.sample_count != 1 {
        return Support::Unsupported("WebGL2 multisample texture storage");
    }
    if desc.dimension == GlTextureDimension::D2Array && desc.sample_count != 1 {
        return Support::Unsupported("GL multisample texture array storage");
    }
    if desc.dimension == GlTextureDimension::D2Array && desc.format.compressed_info().is_some() {
        return Support::Unsupported("GL compressed texture array upload");
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    if !browser && !crate::backend::gl::native::native_texture_format_supported(desc.format) {
        return Support::Unsupported("GL native texture storage mapping");
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    if browser && !crate::backend::gl::browser::supports_texture_storage(desc.format) {
        return Support::Unsupported("WebGL2 texture storage mapping");
    }
    let limits = discovery.limits();
    if desc.extent.width > limits.max_texture_size
        || desc.extent.height > limits.max_texture_size
        || desc.extent.depth_or_layers > limits.max_array_texture_layers
        || desc.extent.width > i32::MAX as u32
        || desc.extent.height > i32::MAX as u32
        || desc.extent.depth_or_layers > i32::MAX as u32
        || desc.mip_level_count > i32::MAX as u32
        || desc.sample_count > limits.max_samples.max(1)
    {
        return Support::Unsupported("GL texture size or sample count");
    }
    let max_mips = u32::BITS - desc.extent.width.max(desc.extent.height).leading_zeros();
    if desc.mip_level_count > max_mips || (desc.sample_count > 1 && desc.mip_level_count != 1) {
        return Support::Unsupported("GL texture mip count");
    }
    let Some(facts) = discovery.formats().get(desc.format, desc.sample_count) else {
        return Support::Unsupported("GL texture format/sample evidence");
    };
    if (desc.usage.contains(GlTextureUsage::SAMPLED) && !facts.sampled)
        || (desc.usage.contains(GlTextureUsage::RENDER_ATTACHMENT) && !facts.renderable)
        || (desc.usage.contains(GlTextureUsage::COPY_SOURCE) && !facts.copy_source)
        || (desc.usage.contains(GlTextureUsage::COPY_DESTINATION) && !facts.copy_destination)
        || (desc.usage.contains(GlTextureUsage::STORAGE_BINDING)
            && !(facts.storage_read || facts.storage_write))
    {
        return Support::Unsupported("GL texture format usage evidence");
    }
    Support::Supported(TextureSupport {
        max_extent: [limits.max_texture_size, limits.max_texture_size, 1],
        max_array_layers: limits.max_array_texture_layers,
        max_mips,
        sample_counts: vec![desc.sample_count],
        compatible_view_formats: Vec::new(),
        filterable: facts.filterable,
        blendable: facts.blendable,
        storage: facts.storage_read || facts.storage_write,
    })
}

/// Converts only the addressing modes that every adopted GL profile exposes.
/// Border and zero addressing need texture-border state or shader emulation;
/// neither is part of the direct 02 route.
fn lower_address(value: api::AddressMode) -> Result<crate::backend::gl::api::GlAddressMode> {
    match value {
        api::AddressMode::ClampToEdge => Ok(crate::backend::gl::api::GlAddressMode::ClampToEdge),
        api::AddressMode::Repeat => Ok(crate::backend::gl::api::GlAddressMode::Repeat),
        api::AddressMode::MirrorRepeat => {
            Ok(crate::backend::gl::api::GlAddressMode::MirroredRepeat)
        }
        api::AddressMode::ClampToBorder | api::AddressMode::ClampToZero => {
            Err(Error::Unsupported("GL sampler address mode"))
        }
    }
}

fn lower_filter(value: api::FilterMode) -> crate::backend::gl::api::GlFilterMode {
    match value {
        api::FilterMode::Nearest => crate::backend::gl::api::GlFilterMode::Nearest,
        api::FilterMode::Linear => crate::backend::gl::api::GlFilterMode::Linear,
    }
}

fn lower_mip_filter(value: api::FilterMode) -> crate::backend::gl::api::GlMipmapFilterMode {
    match value {
        api::FilterMode::Nearest => crate::backend::gl::api::GlMipmapFilterMode::Nearest,
        api::FilterMode::Linear => crate::backend::gl::api::GlMipmapFilterMode::Linear,
    }
}

fn lower_compare(value: api::CompareFunction) -> crate::backend::gl::api::GlCompareFunction {
    match value {
        api::CompareFunction::Never => crate::backend::gl::api::GlCompareFunction::Never,
        api::CompareFunction::Less => crate::backend::gl::api::GlCompareFunction::Less,
        api::CompareFunction::Equal => crate::backend::gl::api::GlCompareFunction::Equal,
        api::CompareFunction::LessEqual => crate::backend::gl::api::GlCompareFunction::LessEqual,
        api::CompareFunction::Greater => crate::backend::gl::api::GlCompareFunction::Greater,
        api::CompareFunction::NotEqual => crate::backend::gl::api::GlCompareFunction::NotEqual,
        api::CompareFunction::GreaterEqual => {
            crate::backend::gl::api::GlCompareFunction::GreaterEqual
        }
        api::CompareFunction::Always => crate::backend::gl::api::GlCompareFunction::Always,
    }
}
