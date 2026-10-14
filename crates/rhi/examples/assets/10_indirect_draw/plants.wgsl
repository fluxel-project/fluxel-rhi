// WebGPU source for example 10's instanced plant pass.
//
// The original GLSL multiplies positions and normals as row vectors.  WGSL
// uses column vectors, so `transpose(rotation)` preserves that source ABI.

struct Frame {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var albedo_array: texture_2d_array<f32>;
@group(0) @binding(2) var linear_sampler: sampler;

struct PlantIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) instance_position: vec3<f32>,
    @location(5) rotation: vec3<f32>,
    @location(6) scale: f32,
    @location(7) layer: i32,
};

struct PlantOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) light: vec3<f32>,
    @location(4) @interpolate(flat) layer: i32,
};

@vertex
fn plants_vs(input: PlantIn) -> PlantOut {
    let sx = sin(input.rotation.x);
    let cx = cos(input.rotation.x);
    let sy = sin(input.rotation.y);
    let cy = cos(input.rotation.y);
    let sz = sin(input.rotation.z);
    let cz = cos(input.rotation.z);
    let rotate_x = mat4x4<f32>(
        vec4<f32>(cx, sx, 0.0, 0.0), vec4<f32>(-sx, cx, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    let rotate_y = mat4x4<f32>(
        vec4<f32>(cy, 0.0, sy, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(-sy, 0.0, cy, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    let rotate_z = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, cz, sz, 0.0),
        vec4<f32>(0.0, -sz, cz, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    let rotation = rotate_z * rotate_y * rotate_x;
    let world = transpose(rotation) * vec4<f32>(
        input.position * input.scale + input.instance_position, 1.0);

    var output: PlantOut;
    output.position = frame.projection * frame.view * world;
    output.normal = transpose(mat3x3<f32>(rotation[0].xyz, rotation[1].xyz, rotation[2].xyz)) * input.normal;
    output.color = input.color;
    output.uv = input.uv;
    output.light = vec3<f32>(0.0, -5.0, 0.0) - world.xyz;
    output.layer = input.layer;
    return output;
}

@fragment
fn plants_ps(input: PlantOut) -> @location(0) vec4<f32> {
    let sampled = textureSample(albedo_array, linear_sampler, input.uv, input.layer);
    if (sampled.a < 0.5) {
        discard;
    }
    let diffuse = max(dot(normalize(input.normal), normalize(input.light)), 0.0) * input.color;
    return vec4<f32>((vec3<f32>(0.65) + diffuse) * sampled.rgb, 1.0);
}
