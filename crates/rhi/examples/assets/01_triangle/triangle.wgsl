// Indexed triangle after the same fixed source-camera transform as the DXIL
// and SPIR-V example variants. Framebuffer coordinates follow WebGPU.
struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
};

@vertex fn vs_main(
    @location(0) position: vec3f,
    @location(1) color: vec3f,
) -> VertexOut {
    var output: VertexOut;
    output.position = vec4f(position, 1.0);
    output.color = color;
    return output;
}

@fragment fn ps_main(input: VertexOut) -> @location(0) vec4f {
    return vec4f(input.color, 1.0);
}
