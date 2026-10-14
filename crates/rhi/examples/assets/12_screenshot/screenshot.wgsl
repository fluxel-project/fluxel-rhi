// WebGPU source for the screenshot dragon.  The uniform member order is
// intentionally `projection, view, model`: `camera_bytes` keeps the original
// C++ bytes (`projection, model, view`), so `view` receives identity and
// `model` receives the camera transform just as the native shader artifacts do.
struct Transform {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> transform: Transform;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) view_vec: vec3<f32>,
    @location(3) light_vec: vec3<f32>,
};

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    let model_position = transform.model * vec4<f32>(input.position, 1.0);
    let view_position = transform.view * model_position;

    var output: VertexOut;
    output.position = transform.projection * view_position;
    // This deliberately uses the `model` member.  With the preserved C++ UBO
    // ordering it is the camera rotation, matching the DXIL and SPIR-V paths.
    output.normal = mat3x3<f32>(
        transform.model[0].xyz,
        transform.model[1].xyz,
        transform.model[2].xyz,
    ) * input.normal;
    output.color = input.color.rgb;
    output.view_vec = -view_position.xyz;
    // Model coordinates are already WebGPU-style; presentation-axis handling
    // belongs to the backend viewport, so the light stays in model space.
    output.light_vec = vec3<f32>(1.0, 1.0, 1.0) - view_position.xyz;
    return output;
}

@fragment
fn ps_main(input: VertexOut) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal);
    let light = normalize(input.light_vec);
    let view = normalize(input.view_vec);
    let reflected = reflect(-light, normal);
    let ambient = vec3<f32>(0.1);
    let diffuse = vec3<f32>(max(dot(normal, light), 0.0));
    let specular = vec3<f32>(pow(max(dot(reflected, view), 0.0), 16.0) * 0.75);
    return vec4<f32>((ambient + diffuse) * input.color + specular, 1.0);
}
