use super::*;

pub(super) fn array<'a>(value: &'a Value, name: &str) -> Result<&'a Vec<Value>, GltfError> {
    value
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| GltfError::new(format!("glTF has no {name} array")))
}
pub(super) fn array_index<'a>(
    values: &'a [Value],
    index: usize,
    name: &str,
) -> Result<&'a Value, GltfError> {
    values
        .get(index)
        .ok_or_else(|| GltfError::new(format!("{name} index {index} is out of range")))
}
pub(super) fn index(value: &Value) -> Result<usize, GltfError> {
    value
        .as_u64()
        .map(|value| value as usize)
        .ok_or_else(|| GltfError::new("glTF index is not unsigned"))
}
pub(super) fn number(value: &Value, name: &str) -> Result<u64, GltfError> {
    value
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| GltfError::new(format!("missing numeric {name}")))
}
pub(super) fn string<'a>(value: &'a Value, name: &str) -> Result<&'a str, GltfError> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| GltfError::new(format!("missing string {name}")))
}
