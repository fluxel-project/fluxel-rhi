// Browser-WebGPU translation of the textured tunnel shader.
struct Scene {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    model: mat4x4<f32>,
    view_pos: vec4<f32>,
    lod_bias: f32,
};

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var albedo: texture_2d<f32>;
@group(0) @binding(2) var sampler0: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) lod_bias: f32,
    @location(2) normal: vec3<f32>,
    @location(3) view_vector: vec3<f32>,
    @location(4) light_vector: vec3<f32>,
};

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
) -> VertexOut {
    let world = scene.model * vec4<f32>(position, 1.0);
    let normal_matrix = mat3x3<f32>(
        scene.model[0].xyz,
        scene.model[1].xyz,
        scene.model[2].xyz,
    );
    var output: VertexOut;
    output.position = scene.projection * scene.view * world;
    output.uv = uv * vec2<f32>(2.0, 1.0);
    output.lod_bias = scene.lod_bias;
    output.normal = normal_matrix * normal;
    output.light_vector = world.xyz - vec3<f32>(-30.0, 0.0, 0.0);
    output.view_vector = scene.view_pos.xyz - world.xyz;
    return output;
}

@fragment fn ps_main(input: VertexOut) -> @location(0) vec4<f32> {
    let color = textureSampleBias(albedo, sampler0, input.uv, input.lod_bias);
    let normal = normalize(input.normal);
    let light = normalize(input.light_vector);
    let view = normalize(input.view_vector);
    let reflection = reflect(light, normal);
    let diffuse = vec3<f32>(max(dot(normal, light), 0.65));
    let specular = pow(max(dot(reflection, view), 0.0), 16.0) * color.a;
    return vec4<f32>(diffuse * color.rgb + vec3<f32>(specular), 1.0);
}
