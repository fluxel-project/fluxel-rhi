struct Value {
    padding: vec4<u32>,
    value: u32,
    result: u32,
    counter: atomic<u32>,
    tail: u32,
}
@group(0) @binding(0) var<storage, read_write> values: binding_array<Value, 2>;
@group(0) @binding(2) var<storage, read_write> output: array<u32, 2>;

@compute @workgroup_size(2)
fn main(@builtin(local_invocation_index) i: u32) {
    let value = values[i].value;
    values[i].result = value + 100u;
    let old = atomicAdd(&values[i].counter, 1u);
    output[i] = value + old;
}
