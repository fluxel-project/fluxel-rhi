// Fullscreen display pass for presenting a selected compute output.
@group(0) @binding(1) var image: texture_2d<f32>;
@group(0) @binding(2) var image_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
) -> VertexOut {
    var output: VertexOut;
    output.position = vec4<f32>(position, 1.0);
    output.uv = uv;
    return output;
}

@fragment fn ps_main(input: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(image, image_sampler, input.uv);
}
