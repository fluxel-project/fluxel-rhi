#version 460
#extension GL_EXT_shader_atomic_float2 : require

layout(local_size_x = 37, local_size_y = 1, local_size_z = 1) in;

shared float counter;

layout(set = 0, binding = 0, std430) buffer Output {
    float value;
} output_data;

void main() {
    if (gl_LocalInvocationIndex == 0) {
        counter = -37.0;
    }
    barrier();

    atomicMax(counter, float(gl_LocalInvocationIndex + 1));
    barrier();

    if (gl_LocalInvocationIndex == 0) {
        output_data.value = counter;
    }
}
