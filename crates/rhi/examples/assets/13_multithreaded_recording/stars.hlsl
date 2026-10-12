// Direct HLSL port of Vulkan/examples/multithreading/starsphere.*.
#ifdef FLUXEL_VULKAN
struct Push { float4x4 mvp; };
[[vk::push_constant]] Push push;
#else
cbuffer PushBuffer : register(b1, space0) { float4x4 mvp; }
#endif

struct VertexInput { float3 position : LOCATION0; };
struct VertexOutput { float4 position : SV_Position; float3 uvw : TEXCOORD0; };
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    // The C++ vkglTF path applies FlipY before this shader sees the sphere.
    // Fluxel's common coordinate convention keeps the mesh unflipped and
    // Vulkan flips only the viewport, so preserve the source procedural
    // coordinate with this one local reflection.
    output.uvw = float3(input.position.x, -input.position.y, input.position.z);
#ifdef FLUXEL_VULKAN
    output.position = mul(push.mvp, float4(input.position, 1.0));
#else
    output.position = mul(mvp, float4(input.position, 1.0));
#endif
    return output;
}
float hash33(float3 p3) {
    p3 = frac(p3 * float3(443.897, 441.423, 437.195));
    p3 += dot(p3, p3.yxz + float3(19.19, 19.19, 19.19));
    return frac((p3.x + p3.y) * p3.z + (p3.x + p3.z) * p3.y + (p3.y + p3.z) * p3.x);
}
float4 ps_main(VertexOutput input) : SV_TARGET {
    float threshold = 0.99;
    float random = hash33(input.uvw);
    float star = random >= threshold ? pow((random - threshold) / (1.0 - threshold), 16.0) : 0.0;
    float3 atmosphere = saturate(float3(0.1, 0.15, 0.4) * (input.uvw.y + 0.25));
    return float4(star.xxx + atmosphere, 1.0);
}
