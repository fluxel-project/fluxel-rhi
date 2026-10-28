#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require

layout(local_size_x = 37, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Data {
    uint64_t value;
} data;

void main() {
    atomicMin(data.value, uint64_t(gl_GlobalInvocationID.x));
}
