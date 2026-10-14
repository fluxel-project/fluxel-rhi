// WebGPU source for example 10's camera-centred sky sphere.

struct Frame {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;

struct SkyIn {
    @location(0) position: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct SkyOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn sky_vs(input: SkyIn) -> SkyOut {
    var output: SkyOut;
    // A zero w removes the view translation while retaining its rotation.
    let rotated = frame.view * vec4<f32>(input.position, 0.0);
    output.position = frame.projection * vec4<f32>(rotated.xyz, 1.0);
    output.uv = input.uv;
    return output;
}

@fragment
fn sky_ps(input: SkyOut) -> @location(0) vec4<f32> {
    let factor = min(0.5 - (input.uv.y + 0.05), 0.5) / 0.15 + 0.5;
    return mix(vec4<f32>(0.93, 0.9, 0.81, 1.0), vec4<f32>(0.35, 0.5, 1.0, 1.0), factor);
}
