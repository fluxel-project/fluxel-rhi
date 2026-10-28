//! Direct browser resource construction, based on the historical WebGPU lowering.
//!
//! Resource creation is synchronous browser descriptor construction. Native JS
//! objects remain inside typed Core handles; cross-device validation happens in
//! Core before descriptor relationships reach this module.

use fluxel_rhi_core::api::{self, Error, Result, Support};
use js_sys::{Array, Object};
use std::{cell::Cell, rc::Rc};
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
    /// Actual `GPUBufferUsage` bits after WebGPU-specific lowering.
    pub(super) native_usage: u32,
    pub(super) mapped: Rc<Cell<bool>>,
}

#[derive(Clone)]
pub struct WebGpuTexture {
    pub(super) raw: JsValue,
    pub(super) desc: api::TextureDesc,
    pub(super) native_transient_usage: bool,
}

#[derive(Clone)]
pub struct WebGpuTextureView {
    pub(super) raw: JsValue,
    pub(super) _texture: WebGpuTexture,
    pub(super) desc: api::TextureViewDesc,
}

#[derive(Clone)]
pub struct WebGpuSampler {
    pub(super) raw: JsValue,
    pub(super) desc: api::SamplerDesc,
}

#[derive(Clone)]
pub struct WebGpuBindGroupLayout {
    pub(super) raw: JsValue,
    pub(super) entries: Vec<api::BindGroupLayoutEntry>,
}

#[derive(Clone)]
pub struct WebGpuBindGroup {
    pub(super) raw: JsValue,
    /// Sources stay live with a bind group and, after binding, in encoder
    /// submission retention through queue completion.
    pub(super) video_sources: Vec<JsValue>,
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
pub(super) fn native_buffer_usage(usage: api::BufferUsage) -> Result<u32> {
    // WebGPU mapping bits have exclusive native combinations. The one RHI
    // exception is query resolve readback: it is represented logically as
    // MAP_READ but lowered to a non-mappable resolve source plus staging.
    let query_readback = usage.contains(api::BufferUsage::MAP_READ)
        && usage.contains(api::BufferUsage::QUERY_RESOLVE);
    if query_readback {
        let allowed = api::BufferUsage::MAP_READ
            .union(api::BufferUsage::QUERY_RESOLVE)
            .union(api::BufferUsage::COPY_SRC)
            .union(api::BufferUsage::COPY_DST);
        if usage.0 & !allowed.0 != 0 {
            return Err(Error::Unsupported(
                "WebGPU query readback usage combination",
            ));
        }
    }
    if usage.contains(api::BufferUsage::MAP_READ) && !query_readback {
        let allowed = api::BufferUsage::MAP_READ.union(api::BufferUsage::COPY_DST);
        if usage.0 & !allowed.0 != 0 {
            return Err(Error::Unsupported("WebGPU MAP_READ usage combination"));
        }
    }
    if usage.contains(api::BufferUsage::MAP_WRITE) {
        let allowed = api::BufferUsage::MAP_WRITE.union(api::BufferUsage::COPY_SRC);
        if usage.0 & !allowed.0 != 0 {
            return Err(Error::Unsupported("WebGPU MAP_WRITE usage combination"));
        }
    }
    let usage = if usage.contains(api::BufferUsage::INDIRECT) {
        usage.union(api::BufferUsage::COPY_SRC)
    } else {
        usage
    };
    let usage = if usage.contains(api::BufferUsage::QUERY_RESOLVE) {
        usage.union(api::BufferUsage::COPY_DST)
    } else {
        usage
    };
    if query_readback {
        let without_map_read = api::BufferUsage(usage.0 & !api::BufferUsage::MAP_READ.0)
            .union(api::BufferUsage::COPY_SRC);
        return buffer_usage(without_map_read);
    }
    // Ordinary GPU buffers retain native COPY_SRC so byte-granular writes can
    // preserve the untouched parts of boundary words in a temporary buffer.
    // WebGPU forbids that extra usage on MAP_READ buffers.
    let usage = if !usage.contains(api::BufferUsage::MAP_READ)
        && !usage.contains(api::BufferUsage::MAP_WRITE)
        && usage.contains(api::BufferUsage::COPY_DST)
    {
        usage.union(api::BufferUsage::COPY_SRC)
    } else {
        usage
    };
    buffer_usage(usage)
}

pub(super) fn texture_usage(usage: api::TextureUsage) -> Result<u32> {
    let known = [
        (api::TextureUsage::COPY_SRC, 1),
        (api::TextureUsage::COPY_DST, 2),
        (api::TextureUsage::TEXTURE_BINDING, 4),
        (api::TextureUsage::STORAGE_BINDING, 8),
        (api::TextureUsage::RENDER_ATTACHMENT, 16),
        (api::TextureUsage::TRANSIENT_ATTACHMENT, 32),
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

pub(super) async fn probe_transient_attachment_usage(device: &JsValue) -> bool {
    use js_sys::Promise;
    use wasm_bindgen_futures::JsFuture;

    if js::call1(device, "pushErrorScope", &JsValue::from_str("validation")).is_err() {
        return false;
    }
    let created = (|| -> Result<JsValue> {
        let texture = descriptor();
        let size = descriptor();
        field(&size, "width", JsValue::from_f64(1.0))?;
        field(&size, "height", JsValue::from_f64(1.0))?;
        field(&size, "depthOrArrayLayers", JsValue::from_f64(1.0))?;
        field(&texture, "size", size)?;
        field(&texture, "dimension", JsValue::from_str("2d"))?;
        field(&texture, "format", JsValue::from_str("rgba8unorm"))?;
        field(&texture, "mipLevelCount", JsValue::from_f64(1.0))?;
        field(&texture, "sampleCount", JsValue::from_f64(1.0))?;
        field(&texture, "usage", JsValue::from_f64(0x30 as f64))?;
        js::call1(device, "createTexture", &texture).map_err(backend_error)
    })();
    let scope = js::call0(device, "popErrorScope")
        .map(Promise::from)
        .map_err(backend_error);
    let scope = match scope {
        Ok(promise) => JsFuture::from(promise).await.map_err(backend_error),
        Err(error) => Err(error),
    };
    let native_usage = created.ok().and_then(|texture| {
        let usage = js::property(&texture, "usage").ok()?.as_f64()? as u32;
        let _ = js::call0(&texture, "destroy");
        Some(usage)
    });
    matches!(scope, Ok(error) if error.is_null() || error.is_undefined())
        && native_usage.is_some_and(|usage| usage & 0x20 != 0)
}

#[path = "resource_format.rs"]
mod format;
pub(super) use format::texture_format;

impl fluxel_rhi_core::backend::ResourceBackend for WebGpu {
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
        let (filterable, blendable, storage, _, multisample) =
            resource_validation::capabilities(device, desc.format);
        Support::Supported(api::TextureSupport {
            max_extent,
            max_array_layers: max_layers,
            max_mips: resource_validation::max_mips_for(desc),
            sample_counts: if multisample {
                resource_validation::supported_sample_counts(desc)
            } else {
                vec![1]
            },
            compatible_view_formats: std::iter::once(desc.format)
                .chain(
                    desc.view_formats
                        .iter()
                        .copied()
                        .filter(|format| *format != desc.format),
                )
                .collect(),
            filterable,
            blendable,
            storage,
        })
    }

    fn create_buffer(device: &WebGpuDevice, desc: &api::BufferDesc) -> Result<WebGpuBuffer> {
        let object = descriptor();
        let native_size = desc.size.checked_add(3).ok_or(Error::InvalidRange)? & !3;
        field(&object, "size", JsValue::from_f64(native_size as f64))?;
        let native_usage = native_buffer_usage(desc.usage)?;
        field(&object, "usage", JsValue::from_f64(native_usage as f64))?;
        if let Some(label) = &desc.label {
            field(&object, "label", JsValue::from_str(label))?;
        }
        let raw = js::call1(&device.0.device, "createBuffer", &object).map_err(backend_error)?;
        Ok(WebGpuBuffer {
            raw,
            size: desc.size,
            usage: desc.usage,
            native_usage,
            mapped: Rc::new(Cell::new(false)),
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
        let mut native_usage = texture_usage(desc.usage)?;
        if desc.usage.contains(api::TextureUsage::TRANSIENT_ATTACHMENT)
            && !device.0.transient_attachment_usage
        {
            native_usage &= !0x20;
        }
        let format_name = texture_format(desc.format)?;
        if format_name.starts_with("depth") {
            if desc.usage.contains(api::TextureUsage::COPY_SRC) {
                native_usage |= 4;
            }
            if desc.usage.contains(api::TextureUsage::COPY_DST) {
                native_usage |= 16;
            }
        }
        if format_name != "stencil8" && desc.usage.contains(api::TextureUsage::COPY_SRC) {
            native_usage |= 4;
        }
        if desc.dimension != api::TextureDimension::D1
            && desc.usage.contains(api::TextureUsage::COPY_DST)
            && resource_validation::capabilities(device, desc.format).3
        {
            native_usage |= 16;
        }
        field(&object, "usage", JsValue::from_f64(native_usage as f64))?;
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
        let native_transient_usage = desc.usage.contains(api::TextureUsage::TRANSIENT_ATTACHMENT)
            && js::property(&raw, "usage")
                .ok()
                .and_then(|usage| usage.as_f64())
                .is_some_and(|usage| (usage as u32) & 0x20 != 0);
        Ok(WebGpuTexture {
            raw,
            desc: desc.clone(),
            native_transient_usage,
        })
    }

    fn texture_uses_native_transient_attachment(texture: &WebGpuTexture) -> bool {
        texture.native_transient_usage
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
        let combined = matches!(
            desc.format,
            api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8
        );
        if !combined || desc.aspects == (api::TextureAspects::DEPTH | api::TextureAspects::STENCIL)
        {
            field(
                &object,
                "format",
                JsValue::from_str(texture_format(desc.format)?),
            )?;
        }
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
        let aspect = if desc.aspects == api::TextureAspects::DEPTH {
            "depth-only"
        } else if desc.aspects == api::TextureAspects::STENCIL {
            "stencil-only"
        } else {
            "all"
        };
        field(&object, "aspect", JsValue::from_str(aspect))?;
        let raw = js::call1(&texture.raw, "createView", &object).map_err(backend_error)?;
        Ok(WebGpuTextureView {
            raw,
            _texture: texture.clone(),
            desc: desc.clone(),
        })
    }

    fn create_sampler(device: &WebGpuDevice, desc: &api::SamplerDesc) -> Result<WebGpuSampler> {
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
        Ok(WebGpuSampler {
            raw: js::call1(&device.0.device, "createSampler", &object).map_err(backend_error)?,
            desc: desc.clone(),
        })
    }

    fn create_bind_group_layout(
        device: &WebGpuDevice,
        desc: &api::BindGroupLayoutDesc,
    ) -> Result<WebGpuBindGroupLayout> {
        let object = descriptor();
        let entries = Array::new();
        for entry in &desc.entries {
            entries.push(&bindings::binding_layout_entry(
                entry,
                &device.0.wgsl_language_features,
            )?);
        }
        field(&object, "entries", entries.into())?;
        if let Some(label) = &desc.label {
            field(&object, "label", JsValue::from_str(label))?;
        }
        Ok(WebGpuBindGroupLayout {
            raw: js::call1(&device.0.device, "createBindGroupLayout", &object)
                .map_err(backend_error)?,
            entries: desc.entries.clone(),
        })
    }

    fn create_bind_group(
        device: &WebGpuDevice,
        desc: &api::BindGroupDesc<'_, WebGpu>,
    ) -> Result<WebGpuBindGroup> {
        let object = descriptor();
        // SAFETY: `Device::create_bind_group` checks `desc.layout.device` and
        // every entry's device against this device before this SPI call.
        let layout = unsafe { desc.layout.native_ref() };
        field(&object, "layout", layout.raw.clone())?;
        let entries = Array::new();
        let mut video_sources = Vec::new();
        for entry in desc.entries {
            let layout_entry = layout
                .entries
                .iter()
                .find(|candidate| candidate.binding == entry.binding)
                .ok_or(Error::InvalidState)?;
            super::binding_validation::validate(device, layout_entry, &entry.resource)?;
            let native = descriptor();
            field(&native, "binding", JsValue::from_f64(entry.binding as f64))?;
            let resource = match &entry.resource {
                api::BindingResource::Buffer { buffer, range }
                | api::BindingResource::StorageBufferView { buffer, range, .. } => {
                    // SAFETY: `Device::create_bind_group` checked this buffer's
                    // DeviceId. Both Core buffer forms lower to WebGPU's one
                    // `GPUBufferBinding` representation; the explicit storage
                    // view ABI was validated by Core before this SPI call.
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
                api::BindingResource::VideoFrame(frame) => {
                    // SAFETY: Core validated this frame's DeviceId before
                    // entering the backend SPI.
                    let frame = unsafe { frame.native_ref() };
                    video_sources.push(frame.source_root().clone());
                    frame.raw.clone()
                }
                api::BindingResource::Sampler(sampler) => {
                    // SAFETY: `Device::create_bind_group` checked this sampler's
                    // DeviceId before entering the backend SPI.
                    unsafe { sampler.native_ref() }.raw.clone()
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
        Ok(WebGpuBindGroup {
            raw: js::call1(&device.0.device, "createBindGroup", &object).map_err(backend_error)?,
            video_sources,
        })
    }

    fn create_pipeline_layout(
        device: &WebGpuDevice,
        desc: &api::PipelineLayoutDesc<'_, WebGpu>,
    ) -> Result<JsValue> {
        super::immediate::create_layout(device, desc)
    }
}

pub(super) fn device_limit(device: &JsValue, name: &str) -> u64 {
    js::property(device, "limits")
        .and_then(|limits| js::property(&limits, name))
        .ok()
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0) as u64
}
