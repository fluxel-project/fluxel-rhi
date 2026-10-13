// WebGPU translation of rock_vs / rock_ps.  Scene matrices use the same
// column-vector convention as the HLSL `column_major` cbuffer.
struct Scene {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    light: vec4<f32>,
    local_speed: f32,
    global_speed: f32,
    _padding: vec2<f32>,
};

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var albedo: texture_2d_array<f32>;
@group(0) @binding(2) var linear_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) view_vector: vec3<f32>,
    @location(4) light_vector: vec3<f32>,
    @location(5) layer: f32,
};

fn rotation(angles: vec3<f32>) -> mat3x3<f32> {
    let x = angles.x + scene.local_speed;
    let y = angles.y + scene.local_speed;
    let z = angles.z + scene.local_speed;
    // WGSL constructors take columns.  These columns represent the exact
    // row values used by the HLSL matrices under `mul(matrix, vector)`.
    let mx = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, cos(x), -sin(x)),
        vec3<f32>(0.0, sin(x), cos(x)),
    );
    let my = mat3x3<f32>(
        vec3<f32>(cos(y), 0.0, sin(y)),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(-sin(y), 0.0, cos(y)),
    );
    let mz = mat3x3<f32>(
        vec3<f32>(cos(z), -sin(z), 0.0),
        vec3<f32>(sin(z), cos(z), 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    return mz * my * mx;
}

fn global_rotation() -> mat4x4<f32> {
    let a = scene.global_speed;
    return mat4x4<f32>(
        vec4<f32>(cos(a), 0.0, -sin(a), 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(sin(a), 0.0, cos(a), 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}

@vertex fn rock_vs(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) instance_position: vec3<f32>,
    @location(5) instance_rotation: vec3<f32>,
    @location(6) instance_scale: f32,
    @location(7) instance_layer: f32,
) -> VertexOut {
    let local_rotation = rotation(instance_rotation);
    let global = global_rotation();
    let world = local_rotation * position * instance_scale + instance_position;
    let view_position = scene.view * global * vec4<f32>(world, 1.0);
    let normal_matrix = mat3x3<f32>(
        (scene.view * global)[0].xyz,
        (scene.view * global)[1].xyz,
        (scene.view * global)[2].xyz,
    );

    var output: VertexOut;
    output.position = scene.projection * view_position;
    output.normal = normal_matrix * local_rotation * normal;
    output.color = color;
    output.uv = uv;
    output.view_vector = -view_position.xyz;
    output.light_vector = mat3x3<f32>(scene.view[0].xyz, scene.view[1].xyz, scene.view[2].xyz) * scene.light.xyz - view_position.xyz;
    output.layer = instance_layer;
    return output;
}

@fragment fn rock_ps(input: VertexOut) -> @location(0) vec4<f32> {
    let base = textureSample(albedo, linear_sampler, input.uv, i32(input.layer)).rgb * input.color;
    let normal = normalize(input.normal);
    let light = normalize(input.light_vector);
    let view = normalize(input.view_vector);
    let ndl = max(dot(normal, light), 0.0);
    let diffuse = max(ndl, 0.1) * base;
    let specular = select(
        vec3<f32>(0.0),
        pow(max(dot(reflect(-light, normal), view), 0.0), 16.0) * vec3<f32>(0.75) * base.r,
        ndl > 0.0,
    );
    return vec4<f32>(diffuse + specular, 1.0);
}
