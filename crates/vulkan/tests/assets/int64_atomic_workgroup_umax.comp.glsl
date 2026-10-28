#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require

layout(local_size_x = 37, local_size_y = 1, local_size_z = 1) in;

shared uint64_t counter;

layout(set = 0, binding = 0, std430) buffer Output {
    uint64_t value;
} output_data;

void main() {
    if (gl_LocalInvocationIndex == 0) {
        counter = uint64_t(0);
    }
    barrier();

    uint64_t value = gl_LocalInvocationIndex == 36
        ? uint64_t(-1)
        : uint64_t(gl_LocalInvocationIndex + 1);
    atomicMax(counter, value);
    barrier();

    if (gl_LocalInvocationIndex == 0) {
        output_data.value = counter;
    }
}
