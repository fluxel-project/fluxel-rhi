#version 460
#extension GL_AMD_shader_explicit_vertex_parameter : require

layout(location = 0) out vec4 color;

void main() {
    vec2 ij = gl_BaryCoordSmoothAMD;
    // AMD's vertex convention is 0=K, 1=I, 2=J. Reorder to the RHI's
    // conventional [vertex 0, vertex 1, vertex 2] output for this fixture.
    color = vec4(1.0 - ij.x - ij.y, ij.x, ij.y, 1.0);
}
