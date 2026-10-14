// WebGPU source for example 10's ground pass.

struct Frame {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var albedo: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;

struct GroundIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct GroundOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn ground_vs(input: GroundIn) -> GroundOut {
    var output: GroundOut;
    output.position = frame.projection * frame.view * vec4<f32>(input.position, 1.0);
    output.uv = input.uv * 32.0;
    return output;
}

@fragment
fn ground_ps(input: GroundOut) -> @location(0) vec4<f32> {
    return textureSample(albedo, linear_sampler, input.uv);
}
