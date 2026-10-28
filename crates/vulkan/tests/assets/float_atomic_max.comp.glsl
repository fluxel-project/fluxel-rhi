#version 460
#extension GL_EXT_shader_atomic_float2 : require
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) buffer Data { float value; };
void main() { atomicMax(value, 1.0); }
