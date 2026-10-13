// Browser-WebGPU translation of the occlusion-query scene's three pipelines.
struct Transform {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    model: mat4x4<f32>,
    color: vec4<f32>,
    light_pos: vec4<f32>,
    visible: f32,
    _padding: vec3<f32>,
};

@group(0) @binding(0) var<uniform> transform: Transform;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) visible: f32,
    @location(3) view_vector: vec3<f32>,
    @location(4) light_vector: vec3<f32>,
};

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) _uv: vec2<f32>,
    @location(3) color: vec3<f32>,
) -> VertexOut {
    let world = transform.model * vec4<f32>(position, 1.0);
    let normal_matrix = mat3x3<f32>(
        transform.model[0].xyz,
        transform.model[1].xyz,
        transform.model[2].xyz,
    );
    var output: VertexOut;
    output.position = transform.projection * transform.view * world;
    output.normal = normal_matrix * normal;
    output.color = color * transform.color.rgb;
    output.visible = transform.visible;
    output.view_vector = -world.xyz;
    output.light_vector = transform.light_pos.xyz - world.xyz;
    return output;
}

@fragment fn simple_ps(_input: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}

@fragment fn mesh_ps(input: VertexOut) -> @location(0) vec4<f32> {
    if (input.visible <= 0.0) {
        return vec4<f32>(0.1, 0.1, 0.1, 1.0);
    }
    let normal = normalize(input.normal);
    let light = normalize(input.light_vector);
    let view = normalize(input.view_vector);
    let diffuse = max(dot(normal, light), 0.25) * input.color;
    let specular = pow(max(dot(reflect(-light, normal), view), 0.0), 8.0) * 0.75;
    return vec4<f32>(diffuse + vec3<f32>(specular), 1.0);
}

@fragment fn occluder_ps(input: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 0.5);
}
