// WebGPU equivalent of the offscreen HLSL phong path. The original clip
// distance is always zero for the captured scene and is omitted here.
struct Transform {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    model: mat4x4<f32>,
    light_position: vec4<f32>,
};
@group(0) @binding(0) var<uniform> transform: Transform;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) eye_position: vec3<f32>,
    @location(3) light_vector: vec3<f32>,
};

@vertex fn phong_vs(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
) -> VertexOut {
    let world = transform.model * vec4<f32>(position, 1.0);
    let eye = transform.view * world;
    var output: VertexOut;
    output.position = transform.projection * eye;
    output.normal = normal;
    output.color = color;
    output.eye_position = eye.xyz;
    output.light_vector = normalize(transform.light_position.xyz - eye.xyz);
    return output;
}

@fragment fn phong_ps(input: VertexOut) -> @location(0) vec4<f32> {
    let eye = normalize(-input.eye_position);
    let reflected = normalize(reflect(-input.light_vector, input.normal));
    let ambient = vec4<f32>(0.1, 0.1, 0.1, 1.0);
    let diffuse = vec4<f32>(max(dot(input.normal, input.light_vector), 0.0));
    var specular = vec4<f32>(0.0);
    if dot(input.eye_position, input.normal) < 0.0 {
        specular = vec4<f32>(0.5, 0.5, 0.5, 1.0)
            * pow(max(dot(reflected, eye), 0.0), 16.0) * 0.75;
    }
    return (ambient + diffuse) * vec4<f32>(input.color, 1.0) + specular;
}
