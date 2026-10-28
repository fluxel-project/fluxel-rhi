#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require

layout(local_size_x = 37, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Data {
    int64_t value;
    int64_t returned[37];
} data;

void main() {
    uint lane = gl_GlobalInvocationID.x;
    int64_t old = atomicExchange(data.value, int64_t(lane + 1));
    data.returned[lane] = old;
}
