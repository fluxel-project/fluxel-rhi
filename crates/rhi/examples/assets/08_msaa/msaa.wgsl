// Browser-WebGPU translation of the Voyager MSAA shaders.
// Transform data is uploaded as column-major matrices by the shared scene.
struct Transform {
    projection: mat4x4<f32>,
    model: mat4x4<f32>,
    light_position: vec4<f32>,
};

@group(0) @binding(0) var<uniform> transform: Transform;
@group(0) @binding(1) var base_color: texture_2d<f32>;
@group(0) @binding(2) var base_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec3<f32>,
    @location(3) view_vector: vec3<f32>,
    @location(4) light_vector: vec3<f32>,
};

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
) -> VertexOut {
    let world = transform.model * vec4<f32>(position, 1.0);
    let normal_matrix = mat3x3<f32>(
        transform.model[0].xyz,
        transform.model[1].xyz,
        transform.model[2].xyz,
    );
    let light_matrix = normal_matrix * transform.light_position.xyz;

    var output: VertexOut;
    output.position = transform.projection * world;
    output.normal = normalize(normal_matrix * normal);
    output.uv = uv;
    output.color = color;
    output.light_vector = light_matrix - world.xyz;
    output.view_vector = -world.xyz;
    return output;
}

@fragment fn ps_main(input: VertexOut) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal);
    let light_dir = normalize(input.light_vector);
    let view_dir = normalize(input.view_vector);
    let reflected = reflect(-light_dir, normal);
    let color = textureSample(base_color, base_sampler, input.uv).rgb * input.color;
    let diffuse = max(dot(normal, light_dir), 0.15);
    let specular = pow(max(dot(reflected, view_dir), 0.0), 16.0) * 0.75;
    return vec4<f32>(color * diffuse + vec3<f32>(specular), 1.0);
}
