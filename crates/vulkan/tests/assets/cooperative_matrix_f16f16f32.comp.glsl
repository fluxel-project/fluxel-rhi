#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// This fixture uses the AMD Vulkan profile's reported KHR tuple:
// subgroup 16x16x16, F16 A/B, F32 accumulator/result, non-saturating.
layout(local_size_x = 64) in;

layout(set = 0, binding = 0, std430) readonly buffer MatrixA {
    float16_t values[];
} matrix_a;

layout(set = 0, binding = 1, std430) readonly buffer MatrixB {
    float16_t values[];
} matrix_b;

layout(set = 0, binding = 2, std430) writeonly buffer MatrixOut {
    float values[];
} matrix_out;

void main() {
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> c =
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    coopMatLoad(a, matrix_a.values, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(b, matrix_b.values, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    c = coopMatMulAdd(a, b, c);
    coopMatStore(c, matrix_out.values, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
}
