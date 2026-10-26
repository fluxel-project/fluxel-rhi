//! Core resource, view, sampler, and binding-layout lowering.
//!
//! This module owns immutable resource descriptions and direct native object
//! allocation. Bind groups retain resources for lifetime only; applying them
//! remains an immediate raster verb.

use fluxel_rhi_core::api::{
    self, BindingResource, BufferDesc, BufferSupport, Error, Result, Support, TextureDesc,
    TextureSupport,
};
use fluxel_rhi_core::backend::ResourceBackend;

mod binding;
mod sampler_lowering;
mod support;
use sampler_lowering::{lower_address, lower_compare, lower_filter, lower_mip_filter};
#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    feature = "native-gles-egl",
    all(target_arch = "wasm32", feature = "webgl2")
))]
use support::texture_support_from_discovery;
mod view;
pub(crate) use binding::lower_program_bindings;

use super::{Gl, GlDevice, spi::native_error, types::*};

/// Preserves Core's 64-bit range; dynamic offsets are added at bind time.
fn lower_buffer_binding_range(
    start: u64,
    end: u64,
    allocation: u64,
    min_size: u64,
) -> Result<(u64, u64)> {
    let size = end.checked_sub(start).ok_or(Error::InvalidRange)?;
    if end > allocation || size == 0 || size < min_size {
        return Err(Error::InvalidRange);
    }
    Ok((start, size))
}

impl ResourceBackend for Gl {
    fn buffer_support(device: &GlDevice, desc: &BufferDesc) -> Support<BufferSupport> {
        match crate::backend::translate::lower_buffer_desc(desc) {
            Ok(lowered) => {
                #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
                {
                    let max_size = match device
                        .0
                        .with_native("query GL buffer address limit", |provider| {
                            Ok(provider.max_buffer_bytes_for(lowered))
                        }) {
                        Ok(max_size) => max_size,
                        Err(_) => return Support::Unsupported("native GL context unavailable"),
                    };
                    if lowered.size > max_size {
                        return Support::Unsupported("GL buffer exceeds native ABI range");
                    }
                    return Support::Supported(BufferSupport {
                        alignment: 1,
                        max_size,
                    });
                }
                #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
                {
                    let _ = device;
                    if lowered.size > i32::MAX as u64 {
                        return Support::Unsupported("WebGL2 buffer exceeds GLsizei range");
                    }
                    return Support::Supported(BufferSupport {
                        alignment: 1,
                        max_size: i32::MAX as u64,
                    });
                }
                #[cfg(not(any(
                    all(windows, feature = "native-gl-wgl"),
                    feature = "native-gles-egl",
                    all(target_arch = "wasm32", feature = "webgl2")
                )))]
                {
                    let _ = device;
                    Support::Unsupported("GL platform context")
                }
            }
            Err(Error::Unsupported(reason)) => Support::Unsupported(reason),
            Err(_) => Support::Unsupported("invalid GL buffer description"),
        }
    }

    fn texture_support(device: &GlDevice, desc: &TextureDesc) -> Support<TextureSupport> {
        match crate::backend::translate::lower_texture_desc(desc) {
            Ok(lowered) => {
                let mut support = texture_support(device, lowered);
                if let Support::Supported(facts) = &mut support {
                    facts.compatible_view_formats.push(desc.format);
                    #[cfg(any(
                        all(windows, feature = "native-gl-wgl"),
                        feature = "native-gles-egl"
                    ))]
                    for &candidate in &desc.view_formats {
                        if facts.compatible_view_formats.contains(&candidate) {
                            continue;
                        }
                        let Ok(format) =
                            crate::backend::translate::lower_texture_format(candidate)
                        else {
                            continue;
                        };
                        let supported = device
                            .0
                            .with_native("query texture view support", |provider| {
                                use crate::backend::api::GlFamilyApi as _;
                                let mut candidate_desc = lowered;
                                candidate_desc.format = format;
                                Ok(
                                    provider.supports_texture_view_format(lowered.format, format)
                                        && matches!(
                                            texture_support_from_discovery(
                                                provider.discovery(),
                                                candidate_desc,
                                                false
                                            ),
                                            Support::Supported(_)
                                        ),
                                )
                            })
                            .unwrap_or(false);
                        if supported {
                            facts.compatible_view_formats.push(candidate);
                        }
                    }
                }
                support
            }
            Err(Error::Unsupported(reason)) => Support::Unsupported(reason),
            Err(_) => Support::Unsupported("invalid GL texture description"),
        }
    }

    fn create_buffer(device: &GlDevice, desc: &BufferDesc) -> Result<GlBuffer> {
        let desc = crate::backend::translate::lower_buffer_desc(desc)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::api::GlResourceApi as _;

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
        let view_formats = desc.view_formats.clone();
        let desc = crate::backend::translate::lower_texture_desc(desc)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::api::GlResourceApi as _;

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
                view_formats,
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
                    view_formats,
                    browser,
                }));
            }
            #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
            {
                let _ = (device, desc, view_formats);
                Err(Error::Unsupported("GL platform context"))
            }
        }
    }

    fn create_view(
        device: &GlDevice,
        texture: &GlTexture,
        desc: &api::TextureViewDesc,
    ) -> Result<GlTextureView> {
        view::create_view(device, texture, desc)
    }

    fn create_sampler(device: &GlDevice, desc: &api::SamplerDesc) -> Result<GlSampler> {
        let border = desc.address.contains(&api::AddressMode::ClampToBorder);
        let zero = desc.address.contains(&api::AddressMode::ClampToZero);
        if border != desc.border.is_some() {
            return Err(Error::InvalidState);
        }
        if border
            && zero
            && !matches!(
                desc.border,
                Some(api::BorderColor::TransparentBlack | api::BorderColor::ZeroInteger)
            )
        {
            return Err(Error::Unsupported(
                "GL sampler has incompatible border colors",
            ));
        }
        let address = |mode| match mode {
            api::AddressMode::ClampToBorder | api::AddressMode::ClampToZero => {
                Ok(crate::backend::api::GlAddressMode::ClampToEdge)
            }
            mode => lower_address(mode),
        };
        let native = crate::backend::api::GlSamplerDesc {
            address_mode_u: address(desc.address[0])?,
            address_mode_v: address(desc.address[1])?,
            address_mode_w: address(desc.address[2])?,
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
            use crate::backend::api::GlSamplerApi as _;
            let id = device
                .0
                .with_native("create Core GL sampler", |provider| {
                    if border || zero {
                        let color = match desc.border {
                            Some(api::BorderColor::OpaqueBlack) => [0., 0., 0., 1.],
                            Some(api::BorderColor::OpaqueWhite) => [1., 1., 1., 1.],
                            _ => [0.; 4],
                        };
                        provider.create_border_sampler(
                            native,
                            desc.address.map(|mode| {
                                matches!(
                                    mode,
                                    api::AddressMode::ClampToBorder | api::AddressMode::ClampToZero
                                )
                            }),
                            color,
                        )
                    } else {
                        provider.create_sampler(native)
                    }
                })
                .map_err(native_error)?;
            return Ok(GlSampler::from_inner(GlSamplerInner {
                state: device.0.clone(),
                id,
                filtering: desc.mag_filter == api::FilterMode::Linear
                    || desc.min_filter == api::FilterMode::Linear
                    || desc.mip_filter == api::FilterMode::Linear
                    || desc.anisotropy > 1,
                comparison: desc.compare.is_some(),
            }));
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            {
                if border || zero {
                    return Err(Error::Unsupported("WebGL2 sampler border clamp"));
                }
                native
                    .validate_for(&device.0.owner.discovery_snapshot())
                    .map_err(|error| match error {
                        crate::backend::api::GlSamplerValidationError::AnisotropyExtensionUnavailable
                        | crate::backend::api::GlSamplerValidationError::AnisotropyLimitUnavailable
                        | crate::backend::api::GlSamplerValidationError::AnisotropyExceedsLimit => {
                            Error::Unsupported("WebGL2 sampler anisotropy is unavailable")
                        }
                        _ => Error::InvalidRange,
                    })?;
                let browser = device.0.owner.create_sampler(native)?;
                return Ok(GlSampler::from_inner(GlSamplerInner {
                    state: device.0.clone(),
                    browser,
                    filtering: desc.mag_filter == api::FilterMode::Linear
                        || desc.min_filter == api::FilterMode::Linear
                        || desc.mip_filter == api::FilterMode::Linear
                        || desc.anisotropy > 1,
                    comparison: desc.compare.is_some(),
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
        validate_sampled_texture_dimensions(device, &entries)?;
        if entries
            .iter()
            .any(|entry| matches!(entry.count, api::BindingCount::RuntimeSized { .. }))
        {
            let limits = runtime_array_limits(device)?;
            binding::validate_runtime_capacities(&entries, limits)?;
        }
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
            let lower_element = |element: &BindingResource<'_, Gl>| -> Result<GlBoundResource> {
                Ok(match (expected.slot, element) {
                    (
                        GlBindingSlot::UniformBuffer { min_size, .. },
                        BindingResource::Buffer { buffer, range },
                    ) => {
                        // SAFETY: Core validates resource provenance before this
                        // SPI call. The wrapper is borrowed for this synchronous
                        // extraction, then cloned into the returned bind group.
                        let buffer = unsafe { buffer.native_ref() };
                        let (offset, size) = lower_buffer_binding_range(
                            range.start,
                            range.end,
                            buffer.desc.size,
                            min_size,
                        )?;
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
                        GlBindingSlot::StorageBuffer { min_size, .. },
                        BindingResource::Buffer { buffer, range }
                        | BindingResource::StorageBufferView { buffer, range, .. },
                    ) => {
                        // SAFETY: Core retains the borrowed buffer through creation; the clone retains it thereafter.
                        let buffer = unsafe { buffer.native_ref() };
                        let (offset, size) = lower_buffer_binding_range(
                            range.start,
                            range.end,
                            buffer.desc.size,
                            min_size,
                        )?;
                        if !std::rc::Rc::ptr_eq(&device.0, &buffer.state) {
                            return Err(Error::WrongDevice);
                        }
                        GlBoundResource::StorageBuffer {
                            buffer: buffer.clone(),
                            offset,
                            size,
                        }
                    }
                    (
                        GlBindingSlot::SampledTexture {
                            dimension,
                            sample,
                            multisampled,
                        },
                        BindingResource::TextureView(view),
                    ) => {
                        // SAFETY: see the buffer arm; views retain their textures.
                        let view = unsafe { view.native_ref() };
                        let view_dimension = match view.binding_texture().desc.dimension {
                            crate::backend::api::GlTextureDimension::D1 => {
                                api::TextureViewDimension::D1
                            }
                            crate::backend::api::GlTextureDimension::D2 => {
                                api::TextureViewDimension::D2
                            }
                            crate::backend::api::GlTextureDimension::D2Array => {
                                api::TextureViewDimension::D2Array
                            }
                            crate::backend::api::GlTextureDimension::D3 => {
                                api::TextureViewDimension::D3
                            }
                            crate::backend::api::GlTextureDimension::Cube => {
                                api::TextureViewDimension::Cube
                            }
                            crate::backend::api::GlTextureDimension::CubeArray => {
                                api::TextureViewDimension::CubeArray
                            }
                            _ => return Err(Error::Unsupported("GL sampled texture dimension")),
                        };
                        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
                        let view_dimension = if view.cube_mirror.is_some() {
                            api::TextureViewDimension::Cube
                        } else if view.stencil_mirror.is_some() {
                            view.dimension
                        } else {
                            view_dimension
                        };
                        if !std::rc::Rc::ptr_eq(&device.0, &view.texture.state)
                            || dimension != view_dimension
                            || multisampled != (view.binding_texture().desc.sample_count > 1)
                            || !view.full_sampling_view
                            || !view
                                .texture
                                .desc
                                .usage
                                .contains(crate::backend::api::GlTextureUsage::SAMPLED)
                            || !(if view.aspects == api::TextureAspects::STENCIL
                                && matches!(
                                    view.binding_texture().desc.format,
                                    crate::backend::api::GlFormat::Depth24PlusStencil8
                                        | crate::backend::api::GlFormat::Depth32FloatStencil8
                                ) {
                                sample == api::TextureSampleType::Uint
                            } else {
                                sample_type_compatible(sample, view.binding_texture().desc.format)
                            })
                        {
                            return Err(Error::InvalidState);
                        }
                        // Multisample textures are accessed only by explicit
                        // sample-index loads; no filtering sampler participates,
                        // so format filterability is not a bind-group precondition.
                        if sample == api::TextureSampleType::Float && !multisampled {
                            let mut format_shape = view.texture.desc;
                            format_shape.format = view.binding_texture().desc.format;
                            match texture_support(device, format_shape) {
                                Support::Supported(facts) if facts.filterable => {}
                                _ => return Err(Error::InvalidState),
                            }
                        }
                        GlBoundResource::Texture(view.clone())
                    }
                    (
                        GlBindingSlot::StorageTexture {
                            dimension, format, ..
                        },
                        BindingResource::TextureView(view),
                    ) => {
                        // A layered image binds all slices of a true GL texture
                        // view. A single-layer D2 attachment fallback can use the
                        // parent array allocation with its explicit layer instead.
                        let view = unsafe { view.native_ref() };
                        #[cfg(any(
                            all(windows, feature = "native-gl-wgl"),
                            feature = "native-gles-egl"
                        ))]
                        if dimension == api::TextureViewDimension::D2Array
                            && view.view_texture.is_none()
                            && (view.native.array_layer != 0
                                || view.native.layer_count
                                    != view.texture.desc.extent.depth_or_layers)
                        {
                            // A layered image binding addresses every layer of the
                            // underlying name. An attachment-only interval must
                            // not silently expose layers outside its Core view.
                            return Err(Error::Unsupported(
                                "GL layered storage image view requires texture-view support",
                            ));
                        }
                        #[cfg(any(
                            all(windows, feature = "native-gl-wgl"),
                            feature = "native-gles-egl"
                        ))]
                        if dimension == api::TextureViewDimension::D3
                            && view.source_mip != 0
                            && format == api::TextureFormat::Rgba8Unorm
                            && !device
                                .0
                                .with_native("query 3D storage mip support", |provider| {
                                    Ok(provider.supports_storage_3d_nonzero_mip())
                                })
                                .map_err(native_error)?
                        {
                            return Err(Error::Unsupported(
                                "GL runtime 3D RGBA8 storage image nonzero mip",
                            ));
                        }
                        if !std::rc::Rc::ptr_eq(&device.0, &view.texture.state)
                            || dimension != view.dimension
                            || view.mip_count != 1
                            || !view
                                .texture
                                .desc
                                .usage
                                .contains(crate::backend::api::GlTextureUsage::STORAGE_BINDING)
                            || crate::backend::translate::lower_texture_format(format)?
                                != view.binding_texture().desc.format
                        {
                            return Err(Error::InvalidState);
                        }
                        GlBoundResource::StorageTexture(view.clone())
                    }
                    (
                        GlBindingSlot::Sampler {
                            filtering,
                            comparison,
                        },
                        BindingResource::Sampler(sampler),
                    ) => {
                        // SAFETY: see the buffer arm; sampler clones share one
                        // provider allocation through their Rc inner object.
                        let sampler = unsafe { sampler.native_ref() };
                        if !std::rc::Rc::ptr_eq(&device.0, &sampler.state)
                            || comparison != sampler.comparison
                            || (!filtering && sampler.filtering)
                        {
                            return Err(Error::InvalidState);
                        }
                        GlBoundResource::Sampler(sampler.clone())
                    }
                    _ => return Err(Error::InvalidState),
                })
            };
            let resource = match (&entry.resource, expected.count) {
                (BindingResource::Array(values), count)
                    if match count {
                        api::BindingCount::Fixed(count) => values.len() == count as usize,
                        api::BindingCount::RuntimeSized { max_count } => {
                            !values.is_empty() && values.len() <= max_count as usize
                        }
                        api::BindingCount::One => false,
                    } =>
                {
                    let elements = values
                        .iter()
                        .map(|element| match element {
                            Some(value) => lower_element(value).map(Some),
                            None if expected.partially_bound => Ok(None),
                            None => Err(Error::InvalidState),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    GlBoundResource::Array(elements)
                }
                (BindingResource::Array(_), _) => return Err(Error::InvalidRange),
                (value, api::BindingCount::One) => lower_element(value)?,
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
        super::push::validate_ranges(desc.push_constants, super::instance::push_limit(&device.0)?)?;
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
            native: crate::backend::api::GlPipelineLayout { bindings },
            push_ranges: desc.push_constants.to_vec().into(),
        })
    }
}

/// 1D sampled textures are desktop GL objects.  Keep the layout contract
/// explicit instead of deferring the GLES/WebGL failure until shader linking
/// or a draw binds an impossible texture target.
fn validate_sampled_texture_dimensions(
    device: &GlDevice,
    entries: &[GlBindingLayoutEntry],
) -> Result<()> {
    if !entries.iter().any(|entry| {
        matches!(
            entry.slot,
            GlBindingSlot::SampledTexture {
                dimension: api::TextureViewDimension::D1,
                ..
            }
        )
    }) {
        return Ok(());
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::api::GlFamilyApi as _;
        let desktop = device
            .0
            .with_native("validate GL 1D sampled texture", |provider| {
                Ok(matches!(
                    provider.discovery().context().profile(),
                    crate::backend::api::GlFamilyProfile::Desktop { .. }
                ))
            })
            .map_err(native_error)?;
        return desktop.then_some(()).ok_or(Error::Unsupported(
            "GL 1D sampled texture requires desktop GL",
        ));
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        let _ = device;
        return Err(Error::Unsupported(
            "GL 1D sampled texture requires desktop GL",
        ));
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = device;
        Err(Error::Unsupported("GL platform context"))
    }
}

/// Obtains the exact native slot facts before a RuntimeSized layout permits
/// Naga specialization. This intentionally runs at layout creation rather
/// than at pipeline linking, where an enormous advertised max_count would
/// already have expanded private shader IR.
fn runtime_array_limits(device: &GlDevice) -> Result<crate::backend::api::GlLimits> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::api::GlFamilyApi as _;
        return device
            .0
            .with_native("query GL runtime array limits", |provider| {
                Ok(provider.discovery().limits())
            })
            .map_err(native_error);
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        return Ok(device.0.owner.discovery_snapshot().limits());
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = device;
        Err(Error::Unsupported("GL platform context"))
    }
}

/// Reports only format/count/size combinations proven by this context's
/// discovery snapshot. `create_texture` performs the final native allocation;
/// no guessed GL specification maximum is published through Core.
fn texture_support(
    device: &GlDevice,
    desc: crate::backend::api::GlTextureDesc,
) -> Support<TextureSupport> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::api::GlFamilyApi as _;
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

fn sample_type_compatible(
    sample: api::TextureSampleType,
    format: crate::backend::api::GlFormat,
) -> bool {
    use crate::backend::api::GlFormat as F;
    let class = match format {
        F::R8Uint
        | F::Rg8Uint
        | F::Rgba8Uint
        | F::R16Uint
        | F::Rg16Uint
        | F::Rgba16Uint
        | F::Rgb10a2Uint
        | F::R32Uint
        | F::R64Uint
        | F::Rg32Uint
        | F::Rgba32Uint => api::TextureSampleType::Uint,
        F::R8Sint
        | F::Rg8Sint
        | F::Rgba8Sint
        | F::R16Sint
        | F::Rg16Sint
        | F::Rgba16Sint
        | F::R32Sint
        | F::Rg32Sint
        | F::Rgba32Sint => api::TextureSampleType::Sint,
        F::Depth16Unorm
        | F::Depth24Unorm
        | F::Depth24PlusStencil8
        | F::Depth32Float
        | F::Depth32FloatStencil8 => api::TextureSampleType::Depth,
        F::Stencil8 => return false,
        _ => api::TextureSampleType::Float,
    };
    class == sample
        || (class == api::TextureSampleType::Float
            && sample == api::TextureSampleType::UnfilterableFloat)
}

/// Converts only the addressing modes that every adopted GL profile exposes.
/// Border and zero addressing need texture-border state or shader emulation;
/// neither is part of the direct 02 route.
#[cfg(test)]
#[path = "../../../tests/backend/rhi/resource/module__tests.rs"]
mod tests;
