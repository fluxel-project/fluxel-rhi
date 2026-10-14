// Native multiview variant: intentionally not browser-WebGPU WGSL.
struct Eyes {
    projection: array<mat4x4<f32>, 2>,
    modelview: array<mat4x4<f32>, 2>,
    light_position: vec4<f32>,
};

@group(0) @binding(0) var<uniform> eyes: Eyes;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) view_vec: vec3<f32>,
    @location(3) light_vec: vec3<f32>,
    @location(4) uv: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    // Naga 27's multiview ABI exposes this SPIR-V builtin as signed i32.
    @builtin(view_index) view: i32,
) -> VertexOut {
    let eye = u32(view);
    let world = eyes.modelview[eye] * vec4<f32>(position, 1.0);
    var output: VertexOut;
    output.position = eyes.projection[eye] * world;
    output.normal = mat3x3<f32>(
        eyes.modelview[eye][0].xyz,
        eyes.modelview[eye][1].xyz,
        eyes.modelview[eye][2].xyz,
    ) * normal;
    output.color = color;
    output.light_vec = (eyes.modelview[eye] * eyes.light_position).xyz - world.xyz;
    output.view_vec = -world.xyz;
    // Retain the native vertex-layout UV input even though this scene does not
    // sample a texture; the pipeline shares the full glTF vertex declaration.
    output.uv = uv;
    return output;
}

@fragment
fn ps_main(input: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(input.normal);
    let l = normalize(input.light_vec);
    let v = normalize(input.view_vec);
    let reflected = reflect(-l, n);
    let ambient = vec3<f32>(0.1);
    let diffuse = max(dot(n, l), 0.0);
    let specular = pow(max(dot(reflected, v), 0.0), 16.0) * vec3<f32>(0.75);
    return vec4<f32>((ambient + vec3<f32>(diffuse)) * input.color + specular, 1.0);
}
