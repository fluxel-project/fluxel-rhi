#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_image_int64 : require
layout(local_size_x = 37) in;
layout(set = 0, binding = 0, r64ui) uniform u64image2D target_image;

void main() {
    uint64_t value = gl_GlobalInvocationID.x == 0 ? ~uint64_t(0) : uint64_t(gl_GlobalInvocationID.x);
    imageAtomicMax(target_image, ivec2(0, 0), value);
}
