#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) buffer OutputBuffer {
    int64_t value;
} output_buffer;
void main() {
    atomicAdd(output_buffer.value, int64_t(1));
}
