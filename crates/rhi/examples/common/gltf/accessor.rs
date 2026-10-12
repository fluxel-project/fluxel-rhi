use super::*;

pub(super) struct Attribute {
    values: Vec<f32>,
    pub(super) count: usize,
    pub(super) components: usize,
}
impl Attribute {
    pub(super) fn at(&self, vertex: usize, component: usize) -> f32 {
        self.values
            .get(vertex * self.components + component)
            .copied()
            .unwrap_or(match component {
                3 => 1.0,
                _ => 0.0,
            })
    }
}

pub(super) fn optional_f32(
    document: &Value,
    buffers: &[Vec<u8>],
    accessor: Option<&Value>,
    count: usize,
    components: usize,
    normalized: bool,
) -> Result<Attribute, GltfError> {
    match accessor {
        Some(accessor) => {
            let attribute = read_f32(document, buffers, index(accessor)?, normalized)?;
            if attribute.count != count
                || !(attribute.components == components
                    || components == 4 && attribute.components == 3)
            {
                return Err(GltfError::new(
                    "primitive attributes have mismatched counts or dimensions",
                ));
            }
            Ok(attribute)
        }
        None => Ok(Attribute {
            values: Vec::new(),
            count,
            components: 0,
        }),
    }
}

pub(super) fn read_f32(
    document: &Value,
    buffers: &[Vec<u8>],
    accessor: usize,
    force_normalized: bool,
) -> Result<Attribute, GltfError> {
    let raw = accessor_bytes(document, buffers, accessor)?;
    let mut values = Vec::with_capacity(raw.count * raw.components);
    for element in 0..raw.count {
        for component in 0..raw.components {
            values.push(component_f32(
                raw.element(element, component)?,
                raw.component_type,
                raw.normalized || force_normalized,
            )?);
        }
    }
    Ok(Attribute {
        values,
        count: raw.count,
        components: raw.components,
    })
}

pub(super) fn read_indices(
    document: &Value,
    buffers: &[Vec<u8>],
    accessor: usize,
) -> Result<Vec<u32>, GltfError> {
    let raw = accessor_bytes(document, buffers, accessor)?;
    if raw.components != 1 {
        return Err(GltfError::new("index accessor must be SCALAR"));
    }
    (0..raw.count)
        .map(|element| {
            let bytes = raw.element(element, 0)?;
            match raw.component_type {
                5121 => Ok(bytes[0] as u32),
                5123 => Ok(u16::from_le_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| GltfError::new("bad u16 index"))?,
                ) as u32),
                5125 => Ok(u32::from_le_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| GltfError::new("bad u32 index"))?,
                )),
                _ => Err(GltfError::new(
                    "indices must use UNSIGNED_BYTE, UNSIGNED_SHORT, or UNSIGNED_INT",
                )),
            }
        })
        .collect()
}

pub(super) struct RawAccessor<'a> {
    data: &'a [u8],
    start: usize,
    stride: usize,
    count: usize,
    components: usize,
    component_type: u64,
    normalized: bool,
    component_size: usize,
}
impl<'a> RawAccessor<'a> {
    pub(super) fn element(&self, element: usize, component: usize) -> Result<&'a [u8], GltfError> {
        let start = self
            .start
            .checked_add(
                element
                    .checked_mul(self.stride)
                    .ok_or_else(|| GltfError::new("accessor offset overflow"))?,
            )
            .and_then(|offset| offset.checked_add(component * self.component_size))
            .ok_or_else(|| GltfError::new("accessor offset overflow"))?;
        self.data
            .get(start..start + self.component_size)
            .ok_or_else(|| GltfError::new("accessor data is truncated"))
    }
}

pub(super) fn accessor_bytes<'a>(
    document: &'a Value,
    buffers: &'a [Vec<u8>],
    accessor_index: usize,
) -> Result<RawAccessor<'a>, GltfError> {
    let accessor = array_index(array(document, "accessors")?, accessor_index, "accessor")?;
    if accessor.get("sparse").is_some() {
        return Err(GltfError::new(
            "sparse accessors are outside the selected-example loader",
        ));
    }
    let view = array_index(
        array(document, "bufferViews")?,
        index(
            accessor
                .get("bufferView")
                .ok_or_else(|| GltfError::new("accessor has no bufferView"))?,
        )?,
        "bufferView",
    )?;
    let component_type = number(accessor, "componentType")?;
    let component_size = component_size(component_type)?;
    let components = match string(accessor, "type")? {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        _ => {
            return Err(GltfError::new(
                "only scalar and vector accessors are supported",
            ));
        }
    };
    let stride = view
        .get("byteStride")
        .and_then(Value::as_u64)
        .unwrap_or((components * component_size) as u64) as usize;
    let buffer = buffers
        .get(number(view, "buffer")? as usize)
        .ok_or_else(|| GltfError::new("bufferView selects a missing buffer"))?;
    let start = view.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize
        + accessor
            .get("byteOffset")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
    Ok(RawAccessor {
        data: buffer,
        start,
        stride,
        count: number(accessor, "count")? as usize,
        components,
        component_type,
        normalized: accessor
            .get("normalized")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        component_size,
    })
}

pub(super) fn component_f32(bytes: &[u8], ty: u64, normalized: bool) -> Result<f32, GltfError> {
    let value = match ty {
        5126 => {
            return Ok(f32::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| GltfError::new("bad float accessor"))?,
            ));
        }
        5120 => i8::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| GltfError::new("bad i8 accessor"))?,
        ) as f32,
        5121 => bytes[0] as f32,
        5122 => i16::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| GltfError::new("bad i16 accessor"))?,
        ) as f32,
        5123 => u16::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| GltfError::new("bad u16 accessor"))?,
        ) as f32,
        5125 => u32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| GltfError::new("bad u32 accessor"))?,
        ) as f32,
        _ => return Err(GltfError::new("unsupported accessor component type")),
    };
    Ok(if normalized {
        match ty {
            5120 => (value / 127.0).max(-1.0),
            5121 => value / 255.0,
            5122 => (value / 32767.0).max(-1.0),
            5123 => value / 65535.0,
            5125 => value / 4_294_967_295.0,
            _ => value,
        }
    } else {
        value
    })
}
pub(super) fn component_size(ty: u64) -> Result<usize, GltfError> {
    match ty {
        5120 | 5121 => Ok(1),
        5122 | 5123 => Ok(2),
        5125 | 5126 => Ok(4),
        _ => Err(GltfError::new("unsupported accessor component type")),
    }
}
