@group(0) @binding(0) var views: texture_2d_array<f32>;
@group(0) @binding(1) var view_sampler: sampler;

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
    let layer = select(0, 1, input.uv.x >= 0.5);
    var uv = vec2<f32>(fract(input.uv.x * 2.0), input.uv.y);
    var p = uv * 2.0 - vec2<f32>(1.0);
    p /= 1.0 - 0.2 * length(p);
    uv = (p + vec2<f32>(1.0)) * 0.5;
    // Derivative-based sampling is only valid in uniform control flow.
    let sampled = textureSample(views, view_sampler, uv, layer);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return sampled;
}
