#version 460
#extension GL_EXT_shader_atomic_float : require
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) buffer OutputBuffer {
    float value;
} output_buffer;
void main() {
    atomicAdd(output_buffer.value, 1.0);
}
