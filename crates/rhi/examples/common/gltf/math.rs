use super::*;

#[derive(Clone, Copy)]
pub(super) struct Mat4([f32; 16]);
impl Mat4 {
    pub(super) const IDENTITY: Self = Self([
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    pub(super) fn mul(self, other: Self) -> Self {
        let mut out = [0.0; 16];
        for column in 0..4 {
            for row in 0..4 {
                out[column * 4 + row] = (0..4)
                    .map(|k| self.0[k * 4 + row] * other.0[column * 4 + k])
                    .sum();
            }
        }
        Self(out)
    }
    pub(super) fn point(self, v: [f32; 3]) -> [f32; 3] {
        let x = self.0[0] * v[0] + self.0[4] * v[1] + self.0[8] * v[2] + self.0[12];
        let y = self.0[1] * v[0] + self.0[5] * v[1] + self.0[9] * v[2] + self.0[13];
        let z = self.0[2] * v[0] + self.0[6] * v[1] + self.0[10] * v[2] + self.0[14];
        let w = self.0[3] * v[0] + self.0[7] * v[1] + self.0[11] * v[2] + self.0[15];
        if w != 0.0 {
            [x / w, y / w, z / w]
        } else {
            [x, y, z]
        }
    }
    pub(super) fn direction(self, v: [f32; 3]) -> [f32; 3] {
        [
            self.0[0] * v[0] + self.0[4] * v[1] + self.0[8] * v[2],
            self.0[1] * v[0] + self.0[5] * v[1] + self.0[9] * v[2],
            self.0[2] * v[0] + self.0[6] * v[1] + self.0[10] * v[2],
        ]
    }
}
pub(super) fn node_matrix(node: &Value) -> Result<Mat4, GltfError> {
    if let Some(values) = node.get("matrix").and_then(Value::as_array) {
        if values.len() != 16 {
            return Err(GltfError::new("node matrix must contain sixteen values"));
        }
        let mut matrix = [0.0; 16];
        for (index, value) in values.iter().enumerate() {
            matrix[index] = value
                .as_f64()
                .ok_or_else(|| GltfError::new("node matrix must be numeric"))?
                as f32;
        }
        return Ok(Mat4(matrix));
    }
    let translation = vector(node.get("translation"), [0.0, 0.0, 0.0])?;
    let scale = vector(node.get("scale"), [1.0, 1.0, 1.0])?;
    let rotation = vector4(node.get("rotation"), [0.0, 0.0, 0.0, 1.0])?;
    let [x, y, z, w] = rotation;
    let rotation = Mat4([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
        0.0,
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
        0.0,
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]);
    Ok(Mat4([
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        translation[0],
        translation[1],
        translation[2],
        1.0,
    ])
    .mul(rotation)
    .mul(Mat4([
        scale[0], 0.0, 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, 0.0, scale[2], 0.0, 0.0, 0.0, 0.0,
        1.0,
    ])))
}
pub(super) fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if length > 0.0 {
        [value[0] / length, value[1] / length, value[2] / length]
    } else {
        value
    }
}
pub(super) fn vector(value: Option<&Value>, default: [f32; 3]) -> Result<[f32; 3], GltfError> {
    match value {
        None => Ok(default),
        Some(value) => {
            let values = value
                .as_array()
                .ok_or_else(|| GltfError::new("node vector must be an array"))?;
            if values.len() != 3 {
                return Err(GltfError::new("node vector must have three values"));
            }
            Ok([
                values[0]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node vector must be numeric"))?
                    as f32,
                values[1]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node vector must be numeric"))?
                    as f32,
                values[2]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node vector must be numeric"))?
                    as f32,
            ])
        }
    }
}
pub(super) fn vector4(value: Option<&Value>, default: [f32; 4]) -> Result<[f32; 4], GltfError> {
    match value {
        None => Ok(default),
        Some(value) => {
            let values = value
                .as_array()
                .ok_or_else(|| GltfError::new("node quaternion must be an array"))?;
            if values.len() != 4 {
                return Err(GltfError::new("node quaternion must have four values"));
            }
            Ok([
                values[0]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node quaternion must be numeric"))?
                    as f32,
                values[1]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node quaternion must be numeric"))?
                    as f32,
                values[2]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node quaternion must be numeric"))?
                    as f32,
                values[3]
                    .as_f64()
                    .ok_or_else(|| GltfError::new("node quaternion must be numeric"))?
                    as f32,
            ])
        }
    }
}
