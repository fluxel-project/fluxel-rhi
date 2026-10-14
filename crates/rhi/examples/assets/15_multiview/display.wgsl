// Sequential-stereo display pass with one sampled image per eye.
@group(0) @binding(0) var left_eye: texture_2d<f32>;
@group(0) @binding(1) var left_sampler: sampler;
@group(1) @binding(0) var right_eye: texture_2d<f32>;
@group(1) @binding(1) var right_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VertexOut;
    out.position = vec4<f32>(uv * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment fn ps_main(input: VertexOut) -> @location(0) vec4<f32> {
    var uv = vec2<f32>(fract(input.uv.x * 2.0), input.uv.y);
    var p = uv * 2.0 - vec2<f32>(1.0);
    p /= 1.0 - 0.2 * length(p);
    uv = (p + vec2<f32>(1.0)) * 0.5;
    let left = textureSample(left_eye, left_sampler, uv);
    let right = textureSample(right_eye, right_sampler, uv);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return select(left, right, input.uv.x >= 0.5);
}
