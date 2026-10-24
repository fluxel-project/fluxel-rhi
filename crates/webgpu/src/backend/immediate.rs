//! WebGPU immediate-data layout and stage-specific command lowering.
use super::{
    command::{error, field, number, object},
    instance::{WebGpu, WebGpuDevice},
    js, resource,
};
use fluxel_rhi_core::api::{self, Error, Result};
use js_sys::{Array, Object, Uint8Array, WeakMap};
use wasm_bindgen::{JsCast, JsValue};

thread_local! {
    static LAYOUTS: WeakMap = WeakMap::new();
    static PIPELINE_SAMPLES: WeakMap = WeakMap::new();
    static DESCRIPTORS: WeakMap = WeakMap::new();
    static PIPELINES: WeakMap = WeakMap::new();
    static PASSES: WeakMap = WeakMap::new();
}
fn get(map: &WeakMap, key: &JsValue) -> JsValue {
    map.get(key.unchecked_ref::<Object>())
}
fn put(map: &WeakMap, key: &JsValue, value: &JsValue) {
    map.set(key.unchecked_ref::<Object>(), value);
}
pub(super) fn create_layout(
    device: &WebGpuDevice,
    desc: &api::PipelineLayoutDesc<'_, WebGpu>,
) -> Result<JsValue> {
    let mut sizes = [0u32; 3];
    for range in desc.push_constants {
        if range.range.start > range.range.end
            || range.range.start % 4 != 0
            || range.range.end % 4 != 0
        {
            return Err(Error::InvalidRange);
        }
        for stage in &range.stages {
            let i = match stage {
                api::ShaderStage::Vertex => 0,
                api::ShaderStage::Fragment => 1,
                api::ShaderStage::Compute => 2,
                _ => return Err(Error::Unsupported("WebGPU shader stage")),
            };
            sizes[i] = sizes[i].max(range.range.end);
        }
    }
    let fragment = sizes[0].checked_add(3).ok_or(Error::InvalidRange)? & !3;
    let total = fragment
        .checked_add(sizes[1])
        .ok_or(Error::InvalidRange)?
        .max(sizes[2]);
    if total > resource::device_limit(&device.0.device, "maxImmediateSize") as u32 {
        return Err(Error::Unsupported("WebGPU maxImmediateSize"));
    }
    if total > 0
        && !device
            .0
            .wgsl_language_features
            .contains("immediate_address_space")
    {
        return Err(Error::Unsupported("WebGPU immediate_address_space"));
    }
    let descriptor = object();
    let groups = Array::new();
    for group in desc.groups {
        // SAFETY: Core checked each layout's device identity.
        groups.push(&unsafe { group.native_ref() }.raw);
    }
    field(&descriptor, "bindGroupLayouts", groups.into())?;
    if total > 0 {
        field(
            &descriptor,
            "immediateSize",
            number(resource::device_limit(&device.0.device, "maxImmediateSize") as u32),
        )?;
    }
    if let Some(label) = desc.label {
        field(&descriptor, "label", JsValue::from_str(label))?;
    }
    let layout = js::call1(&device.0.device, "createPipelineLayout", &descriptor).map_err(error)?;
    let info = Array::new();
    for x in [0, fragment, 0, sizes[0], sizes[1], sizes[2]] {
        info.push(&number(x));
    }
    LAYOUTS.with(|m| put(m, &layout, &info));
    Ok(layout)
}
pub(super) fn layout_info(layout: &JsValue) -> [u32; 6] {
    let info = LAYOUTS.with(|m| get(m, layout));
    if info.is_undefined() {
        return [0; 6];
    }
    let info = Array::from(&info);
    std::array::from_fn(|i| info.get(i as u32).as_f64().unwrap_or(0.0) as u32)
}
pub(super) fn remember_descriptor(descriptor: &JsValue, info: [u32; 6]) {
    let value = Array::new();
    for x in info {
        value.push(&number(x));
    }
    DESCRIPTORS.with(|m| put(m, descriptor, &value));
}
pub(super) fn associate_descriptor(pipeline: &JsValue, descriptor: &JsValue) {
    let info = DESCRIPTORS.with(|m| get(m, descriptor));
    PIPELINES.with(|m| put(m, pipeline, &info));
}
pub(super) fn associate_pipeline(pipeline: &JsValue, layout: &JsValue) {
    let info = LAYOUTS.with(|m| get(m, layout));
    PIPELINES.with(|m| put(m, pipeline, &info));
}
pub(super) fn raster_info(
    device: &WebGpuDevice,
    layout: &JsValue,
    fragment: Option<(&JsValue, &str)>,
) -> Result<[u32; 6]> {
    let mut info = layout_info(layout);
    if info[4] != 0 {
        let alignment = match fragment {
            Some((module, entry)) => super::immediate_shader::required_alignment(
                &super::subgroup::source(module)?,
                entry,
            )?
            .max(4),
            None => 4,
        };
        info[1] = info[3]
            .checked_add(alignment - 1)
            .ok_or(Error::InvalidRange)?
            & !(alignment - 1);
        if info[1].checked_add(info[4]).ok_or(Error::InvalidRange)?
            > resource::device_limit(&device.0.device, "maxImmediateSize") as u32
        {
            return Err(Error::Unsupported("WebGPU maxImmediateSize"));
        }
    }
    Ok(info)
}
pub(super) fn bind_pipeline(pass: &JsValue, pipeline: &JsValue, compute: bool) -> Result<()> {
    let info = PIPELINES.with(|m| get(m, pipeline));
    let state = PASSES.with(|m| get(m, pass));
    let state = if state.is_undefined() {
        let state = object();
        for key in ["vertex", "fragment", "compute"] {
            field(&state, key, Uint8Array::new_with_length(0).into())?;
        }
        state
    } else {
        state
    };
    field(&state, "info", info.clone())?;
    let values = Array::from(&info);
    for (i, key) in ["vertex", "fragment", "compute"].into_iter().enumerate() {
        if (i == 2) != compute {
            continue;
        }
        let old = js::property(&state, key).map_err(error)?;
        let old = Uint8Array::new(&old);
        let size = values.get((i + 3) as u32).as_f64().unwrap_or(0.0) as u32;
        if size == 0 {
            continue;
        }
        let data = Uint8Array::new_with_length(size.max(old.length()));
        let n = old.length();
        data.set(&old.subarray(0, n), 0);
        field(&state, key, data.clone().into())?;
        let base = values.get(i as u32).as_f64().unwrap_or(0.0) as u32;
        js::call(
            pass,
            "setImmediates",
            &[number(base), data.subarray(0, size).into()],
        )
        .map_err(error)?;
    }
    PASSES.with(|m| put(m, pass, &state));
    Ok(())
}
pub(super) fn write(
    pass: &JsValue,
    stages: api::StageMask,
    offset: u32,
    bytes: &[u8],
) -> Result<()> {
    if offset % 4 != 0 || bytes.len() % 4 != 0 {
        return Err(Error::InvalidRange);
    }
    let state = PASSES.with(|m| get(m, pass));
    if state.is_undefined() {
        return Err(Error::InvalidState);
    }
    let info = Array::from(&js::property(&state, "info").map_err(error)?);
    let flags = [
        api::StageMask::VERTEX,
        api::StageMask::FRAGMENT,
        api::StageMask::COMPUTE,
    ];
    if stages.0 == 0 || stages.0 & !flags.iter().fold(0, |a, b| a | b.0) != 0 {
        return Err(Error::InvalidRange);
    }
    let end = offset
        .checked_add(u32::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?)
        .ok_or(Error::InvalidRange)?;
    for (i, flag) in flags.iter().enumerate() {
        if stages.0 & flag.0 != 0 && end > info.get((i + 3) as u32).as_f64().unwrap_or(0.0) as u32 {
            return Err(Error::InvalidRange);
        }
    }
    let data = Uint8Array::from(bytes);
    for (i, flag) in flags.iter().enumerate() {
        if stages.0 & flag.0 != 0 {
            let base = info.get(i as u32).as_f64().unwrap_or(0.0) as u32;
            let cached = Uint8Array::new(
                &js::property(&state, ["vertex", "fragment", "compute"][i]).map_err(error)?,
            );
            cached.set(&data, offset);
            js::call(
                pass,
                "setImmediates",
                &[number(base + offset), data.clone().into()],
            )
            .map_err(error)?;
        }
    }
    Ok(())
}

pub(super) fn variant_module(
    device: &WebGpuDevice,
    module: &JsValue,
    offset: u32,
    entry: &str,
    subgroup: Option<(&str, u32)>,
) -> Result<JsValue> {
    let mut source = super::subgroup::source(module)?;
    if let Some((entry, size)) = subgroup {
        source = super::subgroup::variant(&source, entry, size)?;
    }
    source = super::immediate_shader::variant(&source, offset, entry)?;
    let desc = object();
    field(&desc, "code", JsValue::from_str(&source))?;
    js::call1(&device.0.device, "createShaderModule", &desc).map_err(error)
}

pub(super) fn transfer(old: &JsValue, new: &JsValue) {
    let state = PASSES.with(|m| get(m, old));
    if !state.is_undefined() {
        PASSES.with(|m| put(m, new, &state));
    }
}
pub(super) fn samples(pipeline: &JsValue) -> Result<u32> {
    PIPELINE_SAMPLES
        .with(|m| get(m, pipeline))
        .as_f64()
        .map(|x| x as u32)
        .ok_or(Error::InvalidState)
}

pub(super) fn remember_samples(pipeline: &JsValue, samples: u32) {
    PIPELINE_SAMPLES.with(|m| put(m, pipeline, &number(samples)));
}
