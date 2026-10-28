#version 460
#extension GL_EXT_fragment_shader_barycentric : require

layout(location = 0) out vec4 color;

void main() {
    color = vec4(gl_BaryCoordEXT, 1.0);
}
