// Browser-WebGPU equivalent of the native root-constant shader ABI.
//
// WebGPU deliberately has no push constants.  Each 32-byte `{ color, offset
// }` record is instead selected from this dynamic uniform binding before its
// corresponding indexed draw.  Group 0 remains the native camera ABI.
struct Camera {
    projection_view: mat4x4<f32>,
    model: mat4x4<f32>,
};

struct DrawData {
    color: vec4<f32>,
    offset: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> draw: DrawData;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) _normal: vec3<f32>,
    @location(2) base_color: vec4<f32>,
) -> VertexOut {
    let local = (camera.model * vec4<f32>(position, 1.0)).xyz;
    var output: VertexOut;
    output.position = camera.projection_view * vec4<f32>(local + draw.offset.xyz, 1.0);
    output.color = base_color.rgb * draw.color.rgb;
    return output;
}

@fragment fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
