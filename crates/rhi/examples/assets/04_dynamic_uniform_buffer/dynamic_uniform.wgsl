// WebGPU translation of dynamic_uniform.hlsl.  Both buffers are 64-byte
// matrices; binding 1 is selected by the command's dynamic offset.
struct Camera {
    vp: mat4x4<f32>,
};

struct Model {
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> model: Model;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
) -> VertexOut {
    var output: VertexOut;
    // Rust writes matrices in column-major order.  This is equivalent to the
    // HLSL `mul(vp, mul(model, position))` contract on every RHI backend.
    output.position = camera.vp * model.model * vec4<f32>(position, 1.0);
    output.color = color;
    return output;
}

@fragment fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
