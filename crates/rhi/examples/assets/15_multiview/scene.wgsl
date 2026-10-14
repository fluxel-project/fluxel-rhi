// Sequential eye fallback for 15_multiview. Each ordinary pass binds one
// eye's matrices and writes its independent 2D color target.
struct Eye {
    projection: mat4x4<f32>,
    modelview: mat4x4<f32>,
    light_position: vec4<f32>,
};
@group(0) @binding(0) var<uniform> eye: Eye;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) view_vec: vec3<f32>,
    @location(3) light_vec: vec3<f32>,
};

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) _uv: vec2<f32>,
    @location(3) color: vec3<f32>,
) -> VertexOut {
    let world = eye.modelview * vec4<f32>(position, 1.0);
    var out: VertexOut;
    out.position = eye.projection * world;
    out.normal = mat3x3<f32>(eye.modelview[0].xyz, eye.modelview[1].xyz, eye.modelview[2].xyz) * normal;
    out.color = color;
    out.light_vec = (eye.modelview * eye.light_position).xyz - world.xyz;
    out.view_vec = -world.xyz;
    return out;
}

@fragment fn ps_main(input: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(input.normal);
    let l = normalize(input.light_vec);
    let v = normalize(input.view_vec);
    let reflected = reflect(-l, n);
    let ambient = vec3<f32>(0.1);
    let diffuse = max(dot(n, l), 0.0);
    let specular = pow(max(dot(reflected, v), 0.0), 16.0) * vec3<f32>(0.75);
    return vec4<f32>((ambient + vec3<f32>(diffuse)) * input.color + specular, 1.0);
}
