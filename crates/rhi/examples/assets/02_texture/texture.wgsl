// Textured quad shader; group 0 matches the scene's public resource layout.
struct Scene {
    projection: mat4x4<f32>,
    model_view: mat4x4<f32>,
    view_pos: vec4f,
    lod_bias: f32,
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var color_texture: texture_2d<f32>;
@group(0) @binding(2) var color_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) lod_bias: f32,
    @location(2) normal: vec3f,
    @location(3) view_vec: vec3f,
    @location(4) light_vec: vec3f,
};

@vertex fn vs_main(
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
    @location(2) normal: vec3f,
) -> VertexOut {
    let eye = scene.model_view * vec4f(position, 1.0);
    let normal_matrix = mat3x3<f32>(
        scene.model_view[0].xyz,
        scene.model_view[1].xyz,
        scene.model_view[2].xyz,
    );
    var output: VertexOut;
    output.position = scene.projection * eye;
    output.uv = uv;
    output.lod_bias = scene.lod_bias;
    output.normal = normal_matrix * normal;
    output.light_vec = -eye.xyz;
    output.view_vec = scene.view_pos.xyz - eye.xyz;
    return output;
}

@fragment fn fs_main(input: VertexOut) -> @location(0) vec4f {
    let color = textureSampleLevel(color_texture, color_sampler, input.uv, input.lod_bias);
    let normal = normalize(input.normal);
    let light = normalize(input.light_vec);
    let view = normalize(input.view_vec);
    let reflection = reflect(-light, normal);
    let diffuse = max(dot(normal, light), 0.0);
    let specular = pow(max(dot(reflection, view), 0.0), 16.0) * color.a;
    return vec4f(diffuse * color.rgb + vec3f(specular), 1.0);
}
