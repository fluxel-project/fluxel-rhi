//! Immediate push-constant uniform metadata and stage-local state.
//!
//! Naga exposes each GLSL uniform leaf and its byte offset in the original
//! WGSL block. Values are stage-local; no commands or draw calls are recorded.
use fluxel_rhi_core::api::{Error, PushConstantRange, Result, ShaderStage, StageMask};

#[derive(Clone, Copy, Debug)]
pub(crate) enum PushKind {
    Float(u8),
    Sint(u8),
    Uint(u8),
    Bool(u8),
    Matrix { columns: u8, rows: u8 },
}
#[derive(Clone, Debug)]
pub(crate) struct PushUniform {
    pub name: String,
    pub stage: ShaderStage,
    pub offset: u32,
    pub kind: PushKind,
}

pub(crate) fn reflect(
    module: &naga::Module,
    items: &[naga::back::glsl::PushConstantItem],
    stage: ShaderStage,
) -> Result<Vec<PushUniform>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    // Naga's GLSL reflection walks the right leaves but reconstructs their
    // offsets from natural alignment, ignoring explicit WGSL member offsets
    // and array strides. Preserve its GLSL names and use the IR layout.
    fn leaves(
        module: &naga::Module,
        ty: naga::Handle<naga::Type>,
        offset: u32,
        out: &mut Vec<(naga::Handle<naga::Type>, u32)>,
    ) -> Result<()> {
        match &module.types[ty].inner {
            naga::TypeInner::Scalar(_)
            | naga::TypeInner::Vector { .. }
            | naga::TypeInner::Matrix { .. } => out.push((ty, offset)),
            naga::TypeInner::Struct { members, .. } => {
                for member in members {
                    leaves(
                        module,
                        member.ty,
                        offset
                            .checked_add(member.offset)
                            .ok_or(Error::InvalidRange)?,
                        out,
                    )?;
                }
            }
            naga::TypeInner::Array {
                base,
                size: naga::ArraySize::Constant(count),
                stride,
            } => {
                for i in 0..count.get() {
                    leaves(
                        module,
                        *base,
                        offset
                            .checked_add(i.checked_mul(*stride).ok_or(Error::InvalidRange)?)
                            .ok_or(Error::InvalidRange)?,
                        out,
                    )?;
                }
            }
            _ => return Err(Error::Unsupported("GL push constant uniform type")),
        }
        Ok(())
    }
    let mut offsets = Vec::new();
    for (_, global) in module
        .global_variables
        .iter()
        .filter(|(_, global)| global.space == naga::AddressSpace::PushConstant)
    {
        leaves(module, global.ty, 0, &mut offsets)?;
    }
    if offsets.len() != items.len() {
        return Err(Error::Shader(
            "GL push constant reflection leaf mismatch".into(),
        ));
    }
    items
        .iter()
        .zip(offsets)
        .map(|(item, (ty, offset))| {
            if item.ty != ty {
                return Err(Error::Shader(
                    "GL push constant reflection type mismatch".into(),
                ));
            }
            use naga::{ScalarKind as K, TypeInner as T};
            let kind = match module.types[item.ty].inner {
                T::Scalar(s) if s.width == 4 => match s.kind {
                    K::Float => PushKind::Float(1),
                    K::Sint => PushKind::Sint(1),
                    K::Uint => PushKind::Uint(1),
                    K::Bool => PushKind::Bool(1),
                    _ => return Err(Error::Unsupported("GL push constant scalar type")),
                },
                T::Vector { size, scalar } if scalar.width == 4 => match scalar.kind {
                    K::Float => PushKind::Float(size as u8),
                    K::Sint => PushKind::Sint(size as u8),
                    K::Uint => PushKind::Uint(size as u8),
                    K::Bool => PushKind::Bool(size as u8),
                    _ => return Err(Error::Unsupported("GL push constant vector type")),
                },
                T::Matrix {
                    columns,
                    rows,
                    scalar,
                } if scalar.kind == K::Float && scalar.width == 4 => PushKind::Matrix {
                    columns: columns as u8,
                    rows: rows as u8,
                },
                _ => return Err(Error::Unsupported("GL push constant uniform type")),
            };
            Ok(PushUniform {
                name: item.access_path.clone(),
                stage,
                offset,
                kind,
            })
        })
        .collect()
}

pub(crate) fn validate_ranges(ranges: &[PushConstantRange], limit: u32) -> Result<()> {
    for r in ranges {
        if r.stages.is_empty()
            || r.range.start >= r.range.end
            || r.range.start % 4 != 0
            || r.range.end % 4 != 0
            || r.range.end > limit
        {
            return Err(Error::InvalidRange);
        }
        if r.stages.iter().any(|s| {
            !matches!(
                s,
                ShaderStage::Vertex | ShaderStage::Fragment | ShaderStage::Compute
            )
        }) {
            return Err(Error::Unsupported("GL push constant stage"));
        }
    }
    Ok(())
}

pub(crate) fn stage_index(stage: ShaderStage) -> usize {
    match stage {
        ShaderStage::Vertex => 0,
        ShaderStage::Fragment => 1,
        ShaderStage::Compute => 2,
        _ => unreachable!(),
    }
}
pub(crate) fn stage_mask(stage: ShaderStage) -> StageMask {
    match stage {
        ShaderStage::Vertex => StageMask::VERTEX,
        ShaderStage::Fragment => StageMask::FRAGMENT,
        ShaderStage::Compute => StageMask::COMPUTE,
        _ => StageMask::EMPTY,
    }
}
pub(crate) fn validate_write(
    ranges: &[PushConstantRange],
    stages: StageMask,
    offset: u32,
    bytes: &[u8],
) -> Result<()> {
    let allowed = StageMask::VERTEX.0 | StageMask::FRAGMENT.0 | StageMask::COMPUTE.0;
    if stages.0 == 0
        || stages.0 & !allowed != 0
        || bytes.is_empty()
        || offset % 4 != 0
        || bytes.len() % 4 != 0
    {
        return Err(Error::InvalidRange);
    }
    let end = offset
        .checked_add(u32::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?)
        .ok_or(Error::InvalidRange)?;
    for stage in [
        ShaderStage::Vertex,
        ShaderStage::Fragment,
        ShaderStage::Compute,
    ] {
        if stages.0 & stage_mask(stage).0 != 0 {
            // Every byte must be covered by a declared range for that stage.
            for position in (offset..end).step_by(4) {
                if !ranges.iter().any(|r| {
                    r.stages.contains(&stage)
                        && r.range.start <= position
                        && r.range.end >= position + 4
                }) {
                    return Err(Error::InvalidRange);
                }
            }
        }
    }
    Ok(())
}

fn words(item: &PushUniform, bytes: &[Vec<u8>; 3]) -> Result<Vec<u32>> {
    let (columns, rows, stride) = match item.kind {
        PushKind::Float(n) | PushKind::Sint(n) | PushKind::Uint(n) | PushKind::Bool(n) => (1, n, 0),
        PushKind::Matrix { columns, rows } => (columns, rows, if rows == 2 { 8 } else { 16 }),
    };
    let mut result = Vec::with_capacity(columns as usize * rows as usize);
    let storage = &bytes[stage_index(item.stage)];
    for column in 0..u32::from(columns) {
        for row in 0..u32::from(rows) {
            let offset = item
                .offset
                .checked_add(column * stride + row * 4)
                .ok_or(Error::InvalidRange)? as usize;
            let value = storage
                .get(offset..offset + 4)
                .map(|slice| u32::from_ne_bytes(slice.try_into().expect("four bytes")))
                .unwrap_or(0);
            result.push(value);
        }
    }
    Ok(result)
}

pub(crate) fn validate_uniforms(items: &[PushUniform]) -> Result<()> {
    let empty: [Vec<u8>; 3] = std::array::from_fn(|_| Vec::new());
    for item in items {
        words(item, &empty)?;
    }
    Ok(())
}

pub(crate) fn write(
    encoder: &mut super::encoder::GlEncoder,
    stages: StageMask,
    offset: u32,
    data: &[u8],
    compute: bool,
) -> Result<()> {
    let ranges = if compute {
        &encoder
            .compute_pipeline
            .as_ref()
            .ok_or(Error::InvalidState)?
            .layout
            .push_ranges
    } else {
        &encoder
            .raster_pipeline
            .as_ref()
            .ok_or(Error::InvalidState)?
            .layout
            .push_ranges
    }
    .clone();
    let allowed = if compute {
        StageMask::COMPUTE
    } else {
        StageMask::VERTEX | StageMask::FRAGMENT
    };
    if stages.0 & !allowed.0 != 0 {
        return Err(Error::InvalidRange);
    }
    validate_write(&ranges, stages, offset, data)?;
    let start = offset as usize;
    for stage in [
        ShaderStage::Vertex,
        ShaderStage::Fragment,
        ShaderStage::Compute,
    ] {
        if stages.0 & stage_mask(stage).0 != 0 {
            let storage = &mut encoder.push_bytes[stage_index(stage)];
            if storage.len() < start + data.len() {
                storage.resize(start + data.len(), 0);
            }
            storage[start..start + data.len()].copy_from_slice(data);
        }
    }
    if compute {
        upload_compute(encoder)
    } else {
        upload_raster(encoder)
    }
}

pub(crate) fn upload_raster(encoder: &mut super::encoder::GlEncoder) -> Result<()> {
    let pipeline = encoder.raster_pipeline.clone().ok_or(Error::InvalidState)?;
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        use crate::backend::browser::BrowserPushUniformKind as K;
        let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
        let result = (|| {
            for item in &pipeline.push_uniforms {
                let kind = match item.kind {
                    PushKind::Float(lanes) => K::Float { lanes },
                    PushKind::Sint(lanes) => K::Sint { lanes },
                    PushKind::Uint(lanes) => K::Uint { lanes },
                    PushKind::Bool(lanes) => K::Bool { lanes },
                    PushKind::Matrix { columns, rows } => K::Matrix { columns, rows },
                };
                browser.upload_push_uniform(
                    &pipeline.browser_program,
                    &item.name,
                    kind,
                    &words(item, &encoder.push_bytes)?,
                )?;
            }
            Ok(())
        })();
        if result.is_err() {
            encoder.indeterminate = true;
        } else if !pipeline.push_uniforms.is_empty() {
            encoder.issued = true;
        }
        result?;
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    upload_native(encoder, pipeline.program, &pipeline.push_uniforms)?;
    Ok(())
}

pub(crate) fn upload_compute(encoder: &mut super::encoder::GlEncoder) -> Result<()> {
    let pipeline = encoder
        .compute_pipeline
        .clone()
        .ok_or(Error::InvalidState)?;
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    upload_native(encoder, pipeline.program, &pipeline.push_uniforms)?;
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    let _ = pipeline;
    Ok(())
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn upload_native(
    encoder: &mut super::encoder::GlEncoder,
    program: crate::backend::api::ProgramId,
    items: &[PushUniform],
) -> Result<()> {
    use crate::backend::native::push_uniform::GlPushUniformValue as V;
    if items.is_empty() {
        return Ok(());
    }
    let values = items
        .iter()
        .map(|item| {
            let words = words(item, &encoder.push_bytes)?;
            let value = match item.kind {
                PushKind::Float(_) => V::Float(words.into_iter().map(f32::from_bits).collect()),
                PushKind::Sint(_) | PushKind::Bool(_) => {
                    V::Sint(words.into_iter().map(|v| v as i32).collect())
                }
                PushKind::Uint(_) => V::Uint(words),
                PushKind::Matrix { columns, rows } => V::Matrix {
                    columns,
                    rows,
                    values: words.into_iter().map(f32::from_bits).collect(),
                },
            };
            Ok((item.name.clone(), value))
        })
        .collect::<Result<Vec<_>>>()?;
    let result = encoder
        .state
        .with_native("upload GL push constants", |provider| {
            provider.set_push_uniforms(program, &values)
        })
        .map_err(super::spi::native_error);
    if result.is_err() {
        encoder.indeterminate = true;
    } else {
        encoder.issued = true;
    }
    result
}

/// Validate only linked active leaves, since shader structs may contain unused
/// padding that is intentionally outside the public layout's writable ranges.
pub(crate) fn validate_active(
    items: &[PushUniform],
    ranges: &[PushConstantRange],
    mut active: impl FnMut(&str) -> Result<bool>,
) -> Result<()> {
    for item in items {
        if !active(&item.name)? {
            continue;
        }
        let (columns, rows, stride) = match item.kind {
            PushKind::Float(n) | PushKind::Sint(n) | PushKind::Uint(n) | PushKind::Bool(n) => {
                (1, n, 0)
            }
            PushKind::Matrix { columns, rows } => (columns, rows, if rows == 2 { 8 } else { 16 }),
        };
        for column in 0..u32::from(columns) {
            for row in 0..u32::from(rows) {
                let start = item
                    .offset
                    .checked_add(column * stride + row * 4)
                    .ok_or(Error::InvalidRange)?;
                if !ranges.iter().any(|range| {
                    range.stages.contains(&item.stage)
                        && range.range.start <= start
                        && range.range.end >= start + 4
                }) {
                    return Err(Error::Shader(format!(
                        "active push constant {} is outside the {:?} layout ranges",
                        item.name, item.stage
                    )));
                }
            }
        }
    }
    Ok(())
}
