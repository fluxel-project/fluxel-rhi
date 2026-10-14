//! Direct browser resource construction, based on the historical WebGPU lowering.
//!
//! Resource creation is synchronous browser descriptor construction. Native JS
//! objects remain inside typed Core handles; cross-device validation happens in
//! Core before descriptor relationships reach this module.

use fluxel_rhi_core::api::{self, Error, Result, Support};
use js_sys::{Array, Object};
use wasm_bindgen::JsValue;

use super::{
    bindings,
    instance::{WebGpu, WebGpuDevice},
    js, resource_validation,
};

#[derive(Clone)]
pub struct WebGpuBuffer {
    pub(super) raw: JsValue,
    pub(super) size: u64,
    pub(super) usage: api::BufferUsage,
}

#[derive(Clone)]
pub struct WebGpuTexture {
    pub(super) raw: JsValue,
    pub(super) desc: api::TextureDesc,
}

#[derive(Clone)]
pub struct WebGpuTextureView {
    pub(super) raw: JsValue,
    pub(super) _texture: WebGpuTexture,
}

fn backend_error(error: JsValue) -> Error {
    Error::Backend(js::message(&error))
}

pub(super) fn field(object: &JsValue, name: &str, value: JsValue) -> Result<()> {
    js::set(object, name, &value).map_err(backend_error)
}

pub(super) fn descriptor() -> JsValue {
    Object::new().into()
}

fn buffer_usage(usage: api::BufferUsage) -> Result<u32> {
    let known = [
        (api::BufferUsage::MAP_READ, 1),
        (api::BufferUsage::MAP_WRITE, 2),
        (api::BufferUsage::COPY_SRC, 4),
        (api::BufferUsage::COPY_DST, 8),
        (api::BufferUsage::INDEX, 16),
        (api::BufferUsage::VERTEX, 32),
        (api::BufferUsage::UNIFORM, 64),
        (api::BufferUsage::STORAGE, 128),
        (api::BufferUsage::INDIRECT, 256),
        (api::BufferUsage::QUERY_RESOLVE, 512),
    ];
    let mut native = 0;
    let mut mask = 0;
    for (flag, wire) in known {
        mask |= flag.0;
        if usage.contains(flag) {
            native |= wire;
        }
    }
    if usage.0 == 0 || usage.0 & !mask != 0 {
        return Err(Error::Unsupported("WebGPU buffer usage"));
    }
    Ok(native)
}

/// WebGPU forbids `MAP_READ | QUERY_RESOLVE` on one native buffer.  Core's
/// readback contract still expresses that a resolved query buffer is readable;
/// its ReadbackBackend lowering copies it into a private MAP_READ staging
/// buffer.  Give that source the required native COPY_SRC usage here.
fn native_buffer_usage(usage: api::BufferUsage) -> Result<u32> {
    if usage.contains(api::BufferUsage::MAP_READ) && usage.contains(api::BufferUsage::QUERY_RESOLVE)
    {
        let without_map_read = api::BufferUsage(usage.0 & !api::BufferUsage::MAP_READ.0)
            .union(api::BufferUsage::COPY_SRC);
        return buffer_usage(without_map_read);
    }
    buffer_usage(usage)
}

pub(super) fn texture_usage(usage: api::TextureUsage) -> Result<u32> {
    let known = [
        (api::TextureUsage::COPY_SRC, 1),
        (api::TextureUsage::COPY_DST, 2),
        (api::TextureUsage::TEXTURE_BINDING, 4),
        (api::TextureUsage::STORAGE_BINDING, 8),
        (api::TextureUsage::RENDER_ATTACHMENT, 16),
    ];
    let mut native = 0;
    let mut mask = 0;
    for (flag, wire) in known {
        mask |= flag.0;
        if usage.contains(flag) {
            native |= wire;
        }
    }
    if usage.0 == 0 || usage.0 & !mask != 0 {
        return Err(Error::Unsupported("WebGPU texture usage"));
    }
    Ok(native)
}

pub(super) fn texture_format(format: api::TextureFormat) -> Result<&'static str> {
    Ok(match format {
        api::TextureFormat::R8Unorm => "r8unorm",
        api::TextureFormat::Rg8Unorm => "rg8unorm",
        api::TextureFormat::Rgba8Unorm => "rgba8unorm",
        api::TextureFormat::Rgba8UnormSrgb => "rgba8unorm-srgb",
        api::TextureFormat::Bgra8Unorm => "bgra8unorm",
        api::TextureFormat::Bgra8UnormSrgb => "bgra8unorm-srgb",
        api::TextureFormat::Rgba16Float => "rgba16float",
        api::TextureFormat::R32Float => "r32float",
        api::TextureFormat::Depth16Unorm => "depth16unorm",
        api::TextureFormat::Depth24Plus => "depth24plus",
        api::TextureFormat::Depth24PlusStencil8 => "depth24plus-stencil8",
        api::TextureFormat::Depth32Float => "depth32float",
        _ => return Err(Error::Unsupported("WebGPU texture format")),
    })
}

impl api::ResourceBackend for WebGpu {
    fn buffer_support(
        device: &WebGpuDevice,
        desc: &api::BufferDesc,
    ) -> Support<api::BufferSupport> {
        if desc.size == 0
            || desc
                .size
                .checked_add(3)
                .is_none_or(|size| size & !3 > device_limit(&device.0.device, "maxBufferSize"))
            || native_buffer_usage(desc.usage).is_err()
        {
            return Support::Unsupported("WebGPU buffer descriptor");
        }
        Support::Supported(api::BufferSupport {
            alignment: 4,
            max_size: device_limit(&device.0.device, "maxBufferSize"),
        })
    }

    fn texture_support(
        device: &WebGpuDevice,
        desc: &api::TextureDesc,
    ) -> Support<api::TextureSupport> {
        if !resource_validation::valid_texture_desc(device, desc) {
            return Support::Unsupported("WebGPU texture descriptor");
        }
        let (max_extent, max_layers) = match desc.dimension {
            api::TextureDimension::D1 => (
                [
                    device_limit(&device.0.device, "maxTextureDimension1D") as u32,
                    1,
                    1,
                ],
                1,
            ),
            api::TextureDimension::D2 => {
                let limit = device_limit(&device.0.device, "maxTextureDimension2D") as u32;
                (
                    [limit, limit, 1],
                    device_limit(&device.0.device, "maxTextureArrayLayers") as u32,
                )
            }
            api::TextureDimension::D3 => {
                let limit = device_limit(&device.0.device, "maxTextureDimension3D") as u32;
                ([limit, limit, limit], 1)
            }
        };
        let filterable = matches!(
            desc.format,
            api::TextureFormat::R8Unorm
                | api::TextureFormat::Rg8Unorm
                | api::TextureFormat::Rgba8Unorm
                | api::TextureFormat::Rgba8UnormSrgb
                | api::TextureFormat::Bgra8Unorm
                | api::TextureFormat::Bgra8UnormSrgb
        );
        let depth = matches!(
            desc.format,
            api::TextureFormat::Depth16Unorm
                | api::TextureFormat::Depth24Plus
                | api::TextureFormat::Depth24PlusStencil8
                | api::TextureFormat::Depth32Float
        );
        Support::Supported(api::TextureSupport {
            max_extent,
            max_array_layers: max_layers,
            max_mips: resource_validation::max_mips_for(desc),
            sample_counts: resource_validation::supported_sample_counts(desc),
            compatible_view_formats: vec![desc.format],
            filterable,
            blendable: !depth,
            storage: matches!(
                desc.format,
                api::TextureFormat::Rgba8Unorm
                    | api::TextureFormat::Rgba16Float
                    | api::TextureFormat::R32Float
            ),
        })
    }

    fn create_buffer(device: &WebGpuDevice, desc: &api::BufferDesc) -> Result<WebGpuBuffer> {
        let object = descriptor();
        let native_size = desc.size.checked_add(3).ok_or(Error::InvalidRange)? & !3;
        field(&object, "size", JsValue::from_f64(native_size as f64))?;
        field(
            &object,
            "usage",
            JsValue::from_f64(native_buffer_usage(desc.usage)? as f64),
        )?;
        if let Some(label) = &desc.label {
            field(&object, "label", JsValue::from_str(label))?;
        }
        let raw = js::call1(&device.0.device, "createBuffer", &object).map_err(backend_error)?;
        Ok(WebGpuBuffer {
            raw,
            size: desc.size,
            usage: desc.usage,
        })
    }

    fn create_texture(device: &WebGpuDevice, desc: &api::TextureDesc) -> Result<WebGpuTexture> {
        if !resource_validation::valid_texture_desc(device, desc) {
            return Err(Error::Unsupported("WebGPU texture descriptor"));
        }
        let object = descriptor();
        let size = descriptor();
        field(&size, "width", JsValue::from_f64(desc.extent[0] as f64))?;
        field(&size, "height", JsValue::from_f64(desc.extent[1] as f64))?;
        let depth_or_layers = if desc.dimension == api::TextureDimension::D3 {
            desc.extent[2]
        } else {
            desc.array_layers
        };
        field(
            &size,
            "depthOrArrayLayers",
            JsValue::from_f64(depth_or_layers as f64),
        )?;
        field(&object, "size", size)?;
        field(
            &object,
            "dimension",
            JsValue::from_str(match desc.dimension {
                api::TextureDimension::D1 => "1d",
                api::TextureDimension::D2 => "2d",
                api::TextureDimension::D3 => "3d",
            }),
        )?;
        field(
            &object,
            "format",
            JsValue::from_str(texture_format(desc.format)?),
        )?;
        field(
            &object,
            "usage",
            JsValue::from_f64(texture_usage(desc.usage)? as f64),
        )?;
        field(
            &object,
            "mipLevelCount",
            JsValue::from_f64(desc.mip_levels as f64),
        )?;
        field(
            &object,
            "sampleCount",
            JsValue::from_f64(desc.samples as f64),
        )?;
        if !desc.view_formats.is_empty() {
            let formats = Array::new();
            for &format in &desc.view_formats {
                formats.push(&JsValue::from_str(texture_format(format)?));
            }
            field(&object, "viewFormats", formats.into())?;
        }
        let raw = js::call1(&device.0.device, "createTexture", &object).map_err(backend_error)?;
        Ok(WebGpuTexture {
            raw,
            desc: desc.clone(),
        })
    }

    fn create_view(
        _: &WebGpuDevice,
        texture: &WebGpuTexture,
        desc: &api::TextureViewDesc,
    ) -> Result<WebGpuTextureView> {
        if !resource_validation::valid_view_desc(texture, desc) {
            return Err(Error::InvalidRange);
        }
        let object = descriptor();
        field(
            &object,
            "format",
            JsValue::from_str(texture_format(desc.format)?),
        )?;
        field(
            &object,
            "dimension",
            JsValue::from_str(match desc.dimension {
                api::TextureViewDimension::D1 => "1d",
                api::TextureViewDimension::D2 => "2d",
                api::TextureViewDimension::D2Array => "2d-array",
                api::TextureViewDimension::Cube => "cube",
                api::TextureViewDimension::CubeArray => "cube-array",
                api::TextureViewDimension::D3 => "3d",
            }),
        )?;
        field(
            &object,
            "baseMipLevel",
            JsValue::from_f64(desc.mips.start as f64),
        )?;
        field(
            &object,
            "mipLevelCount",
            JsValue::from_f64(
                desc.mips
                    .end
                    .checked_sub(desc.mips.start)
                    .ok_or(Error::InvalidRange)? as f64,
            ),
        )?;
        field(
            &object,
            "baseArrayLayer",
            JsValue::from_f64(desc.layers.start as f64),
        )?;
        field(
            &object,
            "arrayLayerCount",
            JsValue::from_f64(
                desc.layers
                    .end
                    .checked_sub(desc.layers.start)
                    .ok_or(Error::InvalidRange)? as f64,
            ),
        )?;
        let raw = js::call1(&texture.raw, "createView", &object).map_err(backend_error)?;
        Ok(WebGpuTextureView {
            raw,
            _texture: texture.clone(),
        })
    }

    fn create_sampler(device: &WebGpuDevice, desc: &api::SamplerDesc) -> Result<JsValue> {
        if desc.border.is_some() {
            return Err(Error::Unsupported("WebGPU sampler border color"));
        }
        if !(1..=16).contains(&desc.anisotropy) {
            return Err(Error::InvalidRange);
        }
        if desc.anisotropy > 1
            && (desc.mag_filter != api::FilterMode::Linear
                || desc.min_filter != api::FilterMode::Linear
                || desc.mip_filter != api::FilterMode::Linear)
        {
            return Err(Error::InvalidState);
        }
        let object = descriptor();
        for (field_name, mode) in ["addressModeU", "addressModeV", "addressModeW"]
            .into_iter()
            .zip(desc.address)
        {
            field(
                &object,
                field_name,
                JsValue::from_str(match mode {
                    api::AddressMode::ClampToEdge => "clamp-to-edge",
                    api::AddressMode::Repeat => "repeat",
                    api::AddressMode::MirrorRepeat => "mirror-repeat",
                    _ => return Err(Error::Unsupported("WebGPU sampler address mode")),
                }),
            )?;
        }
        for (field_name, mode) in [
            ("magFilter", desc.mag_filter),
            ("minFilter", desc.min_filter),
            ("mipmapFilter", desc.mip_filter),
        ] {
            field(
                &object,
                field_name,
                JsValue::from_str(match mode {
                    api::FilterMode::Nearest => "nearest",
                    api::FilterMode::Linear => "linear",
                }),
            )?;
        }
        field(
            &object,
            "lodMinClamp",
            JsValue::from_f64(desc.lod.start as f64),
        )?;
        field(
            &object,
            "lodMaxClamp",
            JsValue::from_f64(desc.lod.end as f64),
        )?;
        field(
            &object,
            "maxAnisotropy",
            JsValue::from_f64(desc.anisotropy as f64),
        )?;
        if let Some(compare) = desc.compare {
            field(
                &object,
                "compare",
                JsValue::from_str(match compare {
                    api::CompareFunction::Never => "never",
                    api::CompareFunction::Less => "less",
                    api::CompareFunction::Equal => "equal",
                    api::CompareFunction::LessEqual => "less-equal",
                    api::CompareFunction::Greater => "greater",
                    api::CompareFunction::NotEqual => "not-equal",
                    api::CompareFunction::GreaterEqual => "greater-equal",
                    api::CompareFunction::Always => "always",
                }),
            )?;
        }
        if let Some(label) = &desc.label {
            field(&object, "label", JsValue::from_str(label))?;
        }
        js::call1(&device.0.device, "createSampler", &object).map_err(backend_error)
    }

    fn create_bind_group_layout(
        device: &WebGpuDevice,
        desc: &api::BindGroupLayoutDesc,
    ) -> Result<JsValue> {
        let object = descriptor();
        let entries = Array::new();
        for entry in &desc.entries {
            entries.push(&bindings::binding_layout_entry(entry)?);
        }
        field(&object, "entries", entries.into())?;
        if let Some(label) = &desc.label {
            field(&object, "label", JsValue::from_str(label))?;
        }
        js::call1(&device.0.device, "createBindGroupLayout", &object).map_err(backend_error)
    }

    fn create_bind_group(
        device: &WebGpuDevice,
        desc: &api::BindGroupDesc<'_, WebGpu>,
    ) -> Result<JsValue> {
        let object = descriptor();
        // SAFETY: `Device::create_bind_group` checks `desc.layout.device` and
        // every entry's device against this device before this SPI call.
        let layout = unsafe { desc.layout.native_ref() }.clone();
        field(&object, "layout", layout)?;
        let entries = Array::new();
        for entry in desc.entries {
            let native = descriptor();
            field(&native, "binding", JsValue::from_f64(entry.binding as f64))?;
            let resource = match &entry.resource {
                api::BindingResource::Buffer { buffer, range } => {
                    // SAFETY: `Device::create_bind_group` checked this buffer's
                    // DeviceId; this borrow ends before any asynchronous work.
                    let native_buffer = unsafe { buffer.native_ref() };
                    if range.start >= range.end || range.end > native_buffer.size {
                        return Err(Error::InvalidRange);
                    }
                    let value = descriptor();
                    field(&value, "buffer", native_buffer.raw.clone())?;
                    field(&value, "offset", JsValue::from_f64(range.start as f64))?;
                    field(
                        &value,
                        "size",
                        JsValue::from_f64((range.end - range.start) as f64),
                    )?;
                    value
                }
                api::BindingResource::TextureView(view) => {
                    // SAFETY: `Device::create_bind_group` checked this view's
                    // DeviceId before entering the backend SPI.
                    unsafe { view.native_ref() }.raw.clone()
                }
                api::BindingResource::Sampler(sampler) => {
                    // SAFETY: `Device::create_bind_group` checked this sampler's
                    // DeviceId before entering the backend SPI.
                    unsafe { sampler.native_ref() }.clone()
                }
                _ => return Err(Error::Unsupported("WebGPU binding resource")),
            };
            field(&native, "resource", resource)?;
            entries.push(&native);
        }
        field(&object, "entries", entries.into())?;
        if let Some(label) = desc.label {
            field(&object, "label", JsValue::from_str(label))?;
        }
        js::call1(&device.0.device, "createBindGroup", &object).map_err(backend_error)
    }

    fn create_pipeline_layout(
        device: &WebGpuDevice,
        desc: &api::PipelineLayoutDesc<'_, WebGpu>,
    ) -> Result<JsValue> {
        if !desc.push_constants.is_empty() {
            return Err(Error::Unsupported("WebGPU push constants"));
        }
        let object = descriptor();
        let layouts = Array::new();
        for layout in desc.groups {
            // SAFETY: `Device::create_pipeline_layout` checked each group's
            // DeviceId against this device before the backend SPI call.
            layouts.push(unsafe { layout.native_ref() });
        }
        field(&object, "bindGroupLayouts", layouts.into())?;
        if let Some(label) = desc.label {
            field(&object, "label", JsValue::from_str(label))?;
        }
        js::call1(&device.0.device, "createPipelineLayout", &object).map_err(backend_error)
    }
}

pub(super) fn device_limit(device: &JsValue, name: &str) -> u64 {
    js::property(device, "limits")
        .and_then(|limits| js::property(&limits, name))
        .ok()
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0) as u64
}
