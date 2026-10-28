#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require

layout(local_size_x = 37, local_size_y = 1, local_size_z = 1) in;

shared int64_t counter;

layout(set = 0, binding = 0, std430) buffer Output {
    int64_t value;
} output_data;

void main() {
    if (gl_LocalInvocationIndex == 0) {
        counter = int64_t(0);
    }
    barrier();

    atomicXor(counter, int64_t(1) << int64_t(gl_LocalInvocationIndex % 6));
    barrier();

    if (gl_LocalInvocationIndex == 0) {
        output_data.value = counter;
    }
}
