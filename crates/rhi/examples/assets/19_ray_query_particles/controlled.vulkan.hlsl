// Controlled visual subset of NVIDIA Donut rt_particles; see README for differences.
// Copyright (c) 2014-2023 NVIDIA CORPORATION. MIT license, see assets directory.
RaytracingAccelerationStructure scene : register(t0);
[[vk::image_format("rgba8")]] RWTexture2D<float4> outputImage : register(u1);
StructuredBuffer<float4> triangleNormals : register(t2);
struct Particle { float4 centerRadius; float4 colorOpacity; float4 rotation; };
StructuredBuffer<Particle> particles : register(t3);
StructuredBuffer<uint> environmentPixels : register(t5);
StructuredBuffer<uint> smokePixels : register(t6);
cbuffer Frame : register(b4) {
    float4 cameraPosition;
    float4 cameraForward;
    float4 cameraRight;
    float4 cameraUp;
    uint4 dimensions; // width, height, particle count, unused
    uint4 triangleOffsets0;
    uint4 triangleOffsets1;
};
uint triangleOffset(uint instance) {
    return instance < 4 ? triangleOffsets0[instance] : triangleOffsets1[instance - 4];
}
float3 environmentColor(float3 direction) {
    float2 uv = float2(atan2(direction.z, direction.x) / 6.283185307 - 0.25,
                       0.5 - asin(clamp(direction.y, -1, 1)) / 3.141592654);
    uint2 xy = uint2(frac(uv.x) * 2048, clamp(uv.y, 0, 0.999999) * 1024);
    uint packed = environmentPixels[xy.y * 2048 + xy.x];
    return float3(packed & 255, (packed >> 8) & 255, (packed >> 16) & 255) / 255.0;
}
float4 particleLayer(float3 origin, float3 direction, float maxT) {
    // Camera-facing plane approximation. The source uses triangle candidates
    // for primary rays and procedural AABBs for secondary rays.
    float4 result = float4(0, 0, 0, 1);
    for (uint i = 0; i < dimensions.z; ++i) {
        Particle p = particles[i];
        float3 right = cos(p.rotation.x) * cameraRight.xyz + sin(p.rotation.x) * cameraUp.xyz;
        float3 up = -sin(p.rotation.x) * cameraRight.xyz + cos(p.rotation.x) * cameraUp.xyz;
        float denominator = dot(direction, cameraForward.xyz);
        if (abs(denominator) < 1e-6) continue;
        float t = dot(p.centerRadius.xyz - origin, cameraForward.xyz) / denominator;
        if (t <= 0 || t >= maxT) continue;
        float3 delta = origin + direction * t - p.centerRadius.xyz;
        float2 uv = float2(dot(delta, right), -dot(delta, up)) / p.centerRadius.w * 0.5 + 0.5;
        if (any(uv < 0) || any(uv >= 1)) continue;
        uint2 xy = uint2(uv * 512);
        uint packed = smokePixels[xy.y * 512 + xy.x];
        float3 texel = float3(packed & 255, (packed >> 8) & 255, (packed >> 16) & 255) / 255.0;
        texel = lerp(texel / 12.92, pow((texel + 0.055) / 1.055, 2.4), step(0.04045, texel));
        float alpha = (packed >> 24) / 255.0 * p.colorOpacity.a;
        result.rgb += result.a * alpha * p.colorOpacity.rgb * texel;
        result.a *= 1.0 - alpha;
    }
    return result;
}
[numthreads(8, 8, 1)]
void main(uint2 pixel : SV_DispatchThreadID) {
    if (any(pixel >= dimensions.xy)) return;
    float2 uv = (float2(pixel) + 0.5) / float2(dimensions.xy) * 2 - 1;
    uv.y = -uv.y;
    uv.x *= float(dimensions.x) / dimensions.y;
    float3 origin = cameraPosition.xyz;
    float3 direction = normalize(cameraForward.xyz + 0.41421356 * (uv.x * cameraRight.xyz + uv.y * cameraUp.xyz));
    float3 color = 0;
    float attenuation = 1;
    for (uint bounce = 0; bounce < 8; ++bounce) {
        RayDesc ray; ray.Origin = origin; ray.Direction = direction; ray.TMin = 0.001; ray.TMax = 1000;
        RayQuery<RAY_FLAG_CULL_NON_OPAQUE> q;
        q.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, ray);
        while (q.Proceed()) { }
        bool hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
        float t = hit ? q.CommittedRayT() : ray.TMax;
        float4 layer = particleLayer(origin, direction, t);
        color += attenuation * layer.rgb;
        attenuation *= layer.a;
        if (!hit) break;
        uint firstNormal = 3 * (triangleOffset(q.CommittedInstanceID()) + q.CommittedPrimitiveIndex());
        float2 bary = q.CommittedTriangleBarycentrics();
        float3 normal = normalize(triangleNormals[firstNormal].xyz * (1 - bary.x - bary.y)
            + triangleNormals[firstNormal + 1].xyz * bary.x + triangleNormals[firstNormal + 2].xyz * bary.y);
        if (dot(normal, direction) > 0) normal = -normal;
        attenuation *= 0.8;
        origin += direction * t + normal * 0.001;
        direction = reflect(direction, normal);
    }
    color += attenuation * environmentColor(direction);
    // The reference blits its linear RGBA16_FLOAT buffer to an sRGB swapchain.
    color = saturate(color);
    float3 srgb = lerp(12.92 * color, 1.055 * pow(color, 1.0 / 2.4) - 0.055,
                       step(0.0031308, color));
    outputImage[pixel] = float4(srgb, 1);
}
