// WebGPU translation of planet_vs / planet_ps.  The CPU uploads column-major
// projection and view matrices, matching WGSL matrix multiplication directly.
struct Scene {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    light: vec4<f32>,
    local_speed: f32,
    global_speed: f32,
    _padding: vec2<f32>,
};

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var albedo: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) view_vector: vec3<f32>,
    @location(4) light_vector: vec3<f32>,
};

@vertex fn planet_vs(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
) -> VertexOut {
    let view_position = scene.view * vec4<f32>(position, 1.0);
    let normal_matrix = mat3x3<f32>(
        scene.view[0].xyz,
        scene.view[1].xyz,
        scene.view[2].xyz,
    );

    var output: VertexOut;
    output.position = scene.projection * view_position;
    output.normal = normal_matrix * normal;
    output.color = color;
    output.uv = uv;
    output.view_vector = -view_position.xyz;
    output.light_vector = normal_matrix * scene.light.xyz - view_position.xyz;
    return output;
}

@fragment fn planet_ps(input: VertexOut) -> @location(0) vec4<f32> {
    let base = textureSample(albedo, linear_sampler, input.uv).rgb * input.color * 1.5;
    let normal = normalize(input.normal);
    let light = normalize(input.light_vector);
    let view = normalize(input.view_vector);
    let ndl = max(dot(normal, light), 0.0);
    let diffuse = max(ndl, 0.1) * base;
    let specular = select(
        vec3<f32>(0.0),
        pow(max(dot(reflect(-light, normal), view), 0.0), 4.0) * vec3<f32>(0.75) * base.r,
        ndl > 0.0,
    );
    return vec4<f32>(diffuse + specular, 1.0);
}
