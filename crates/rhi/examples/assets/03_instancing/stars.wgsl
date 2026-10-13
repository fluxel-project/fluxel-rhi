// WebGPU translation of star_vs / star_ps.  The fullscreen triangle uses the
// vertex index, so it has no vertex-buffer inputs.
struct StarOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uvw: vec3<f32>,
};

@vertex fn star_vs(@builtin(vertex_index) id: u32) -> StarOut {
    let uv = vec2<f32>(f32((id << 1u) & 2u), f32(id & 2u));
    var output: StarOut;
    output.uvw = vec3<f32>(uv, uv.y);
    output.position = vec4<f32>(uv * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    return output;
}

fn hash(value: vec3<f32>) -> f32 {
    var p = fract(value * vec3<f32>(443.897, 441.423, 437.195));
    p += vec3<f32>(dot(p, vec3<f32>(p.y, p.x, p.z) + vec3<f32>(19.19)));
    return fract((p.x + p.y) * p.z + (p.x + p.z) * p.y + (p.y + p.z) * p.x);
}

@fragment fn star_ps(input: StarOut) -> @location(0) vec4<f32> {
    let random = hash(input.uvw);
    let star = select(0.0, pow((random - 0.99) / 0.01, 16.0), random >= 0.99);
    return vec4<f32>(vec3<f32>(star), 1.0);
}
