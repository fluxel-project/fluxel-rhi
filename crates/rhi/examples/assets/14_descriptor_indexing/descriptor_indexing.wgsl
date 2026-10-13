struct Uniforms {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    model: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var nearest_sampler: sampler;
@group(0) @binding(2) var textures: texture_2d_array<f32>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) texture_index: i32,
};
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) texture_index: i32,
};
@vertex fn vs_main(input: VertexIn) -> VertexOut {
    var output: VertexOut;
    output.position = uniforms.projection * uniforms.view * uniforms.model * vec4<f32>(input.position, 1.0);
    output.uv = input.uv;
    output.texture_index = input.texture_index;
    return output;
}
@fragment fn ps_main(input: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(textures, nearest_sampler, input.uv, input.texture_index);
}
