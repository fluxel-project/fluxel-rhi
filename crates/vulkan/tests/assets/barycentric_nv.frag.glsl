#version 460
#extension GL_NV_fragment_shader_barycentric : require

layout(location = 0) out vec4 color;

void main() {
    color = vec4(gl_BaryCoordNV, 1.0);
}
