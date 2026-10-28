struct Value {
    padding: vec4<u32>,
    value: vec4<u32>,
}

@group(0) @binding(0) var<uniform> values: binding_array<Value, 2>;
@group(0) @binding(2) var<storage, read_write> output: array<u32, 2>;

@compute @workgroup_size(2)
fn main(@builtin(local_invocation_index) i: u32) {
    output[i] = values[i].value.x;
}
