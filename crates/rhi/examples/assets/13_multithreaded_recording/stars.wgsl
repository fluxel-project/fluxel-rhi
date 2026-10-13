struct StarTransform { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> transform: StarTransform;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uvw: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> VertexOut {
    var output: VertexOut;
    output.uvw = vec3<f32>(position.x, -position.y, position.z);
    output.position = transform.mvp * vec4<f32>(position, 1.0);
    return output;
}

fn hash33(input: vec3<f32>) -> f32 {
    var p = fract(input * vec3<f32>(443.897, 441.423, 437.195));
    p += vec3<f32>(dot(p, p.yxz + vec3<f32>(19.19)));
    return fract((p.x + p.y) * p.z + (p.x + p.z) * p.y + (p.y + p.z) * p.x);
}

@fragment
fn ps_main(input: VertexOut) -> @location(0) vec4<f32> {
    let random = hash33(input.uvw);
    let star = select(0.0, pow((random - 0.99) / 0.01, 16.0), random >= 0.99);
    let atmosphere = clamp(vec3<f32>(0.1, 0.15, 0.4) * (input.uvw.y + 0.25), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(vec3<f32>(star) + atmosphere, 1.0);
}
