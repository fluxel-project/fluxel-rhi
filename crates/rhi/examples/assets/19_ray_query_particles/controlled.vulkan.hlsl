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
// The source uses a wrapping linear sampler. These buffers contain mip zero;
// decode each smoke texel before interpolation as for an sRGB texture view.
float4 unpackColor(uint packed) {
    return float4(packed & 255, (packed >> 8) & 255, (packed >> 16) & 255,
                  packed >> 24) / 255.0;
}
float4 loadEnvironment(int2 xy) {
    xy = (xy % int2(2048, 1024) + int2(2048, 1024)) % int2(2048, 1024);
    return unpackColor(environmentPixels[xy.y * 2048 + xy.x]);
}
float4 loadSmoke(int2 xy, uint mip) {
    int size = 512u >> mip;
    xy = (xy % size + size) % size;
    uint offset = (262144 - size * size) / 3 * 4;
    float4 c = unpackColor(smokePixels[offset + xy.y * size + xy.x]);
    c.rgb = lerp(c.rgb / 12.92, pow((c.rgb + 0.055) / 1.055, 2.4), step(0.04045, c.rgb));
    return c;
}
float4 sampleEnvironment(float2 uv) {
    float2 position = uv * float2(2048, 1024) - 0.5;
    int2 p = int2(floor(position));
    float2 f = frac(position);
    return lerp(lerp(loadEnvironment(p), loadEnvironment(p + int2(1, 0)), f.x),
                lerp(loadEnvironment(p + int2(0, 1)), loadEnvironment(p + 1), f.x), f.y);
}
float4 sampleSmokeLevel(float2 uv, uint mip) {
    float size = float(512u >> mip);
    float2 position = uv * size - 0.5;
    int2 p = int2(floor(position));
    float2 f = frac(position);
    return lerp(lerp(loadSmoke(p, mip), loadSmoke(p + int2(1, 0), mip), f.x),
                lerp(loadSmoke(p + int2(0, 1), mip), loadSmoke(p + 1, mip), f.x), f.y);
}
float4 sampleSmoke(float2 uv, float mip) {
    mip = clamp(mip, 0.0, 9.0);
    uint low = uint(floor(mip));
    return lerp(sampleSmokeLevel(uv, low), sampleSmokeLevel(uv, min(low + 1, 9)), frac(mip));
}
float3 environmentColor(float3 direction) {
    float2 uv = float2(atan2(direction.z, direction.x) / 6.283185307 - 0.25,
                       0.5 - asin(clamp(direction.y, -1, 1)) / 3.141592654);
    return sampleEnvironment(uv).rgb;
}
float3 rotateByQuaternion(float4 q, float3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
float4 particleLayer(float3 origin, float3 direction, float maxT, float pathLength, bool secondary) {
    // Primary geometry is evaluated in software; secondary particles use the
    // procedural candidates from the particle instance mask, as in the source.
    // The reference defaults to four MLAB fragments. Each entry stores
    // premultiplied color, transmittance, and ray distance.
    float4 layers[4];
    float depths[4];
    for (uint j = 0; j < 4; ++j) {
        layers[j] = float4(0, 0, 0, 1);
        depths[j] = 1e30;
    }
    RayDesc particleRay;
    particleRay.Origin = origin;
    particleRay.Direction = direction;
    particleRay.TMin = 0;
    particleRay.TMax = maxT;
    RayQuery<RAY_FLAG_NONE> particleQuery;
    if (secondary) particleQuery.TraceRayInline(scene, RAY_FLAG_NONE, 4, particleRay);
    uint i = 0;
    while (secondary ? particleQuery.Proceed() : i < dimensions.z) {
        uint particleIndex;
        if (secondary) {
            if (particleQuery.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) continue;
            particleIndex = particleQuery.CandidateInstanceID();
            if (particleIndex >= dimensions.z) continue;
        } else {
            particleIndex = i++;
        }
        Particle p = particles[particleIndex];
        float3 right = cos(p.rotation.x) * cameraRight.xyz + sin(p.rotation.x) * cameraUp.xyz;
        float3 up = -sin(p.rotation.x) * cameraRight.xyz + cos(p.rotation.x) * cameraUp.xyz;
        float3 normal = normalize(cross(right, up));
        if (secondary) {
            float3 virtualCamera = origin - direction * pathLength;
            float3 facingNormal = normalize(virtualCamera - p.centerRadius.xyz);
            if (dot(normal, facingNormal) < 0.0) normal = -normal;
            float4 q = float4(cross(normal, facingNormal), 1.0 + dot(normal, facingNormal));
            float qLength = length(q);
            q = qLength > 0 ? q / qLength : float4(1, 0, 0, 0);
            right = rotateByQuaternion(q, right);
            up = rotateByQuaternion(q, up);
            normal = facingNormal;
        }
        float denominator = dot(direction, normal);
        if (abs(denominator) < 1e-6) continue;
        float t = dot(p.centerRadius.xyz - origin, normal) / denominator;
        if (t <= 0 || t >= maxT) continue;
        float3 delta = origin + direction * t - p.centerRadius.xyz;
        float2 uv = float2(dot(delta, right), -dot(delta, up)) / p.centerRadius.w * 0.5 + 0.5;
        if (any(uv < 0) || any(uv >= 1)) continue;
        float footprint = (pathLength + t) * (0.7853981634 / 709.0);
        float mip = log2(256.0 * footprint / p.centerRadius.w);
        float4 smoke = sampleSmoke(uv, mip);
        float3 texel = smoke.rgb;
        float alpha = smoke.a * p.colorOpacity.a;
        if (alpha == 0) continue;
        float4 fragment = float4(alpha * p.colorOpacity.rgb * texel, 1.0 - alpha);
        float depth = t;
        for (uint j = 0; j < 4; ++j) {
            if (depth < depths[j]) {
                float4 previous = layers[j];
                float previousDepth = depths[j];
                layers[j] = fragment;
                depths[j] = depth;
                fragment = previous;
                depth = previousDepth;
            }
        }
        layers[3].rgb += fragment.rgb * layers[3].a;
        layers[3].a *= fragment.a;
    }
    float4 result = layers[0];
    for (uint j = 1; j < 4; ++j) {
        result.rgb += layers[j].rgb * result.a;
        result.a *= layers[j].a;
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
    float pathLength = 0;
    for (uint bounce = 0; bounce < 8; ++bounce) {
        RayDesc ray; ray.Origin = origin; ray.Direction = direction; ray.TMin = 0; ray.TMax = 1000;
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES> q;
        q.TraceRayInline(scene, RAY_FLAG_NONE, 1, ray);
        while (q.Proceed()) { }
        bool hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
        float t = hit ? q.CommittedRayT() : ray.TMax;
        float4 layer = particleLayer(origin, direction, t, pathLength, bounce > 0);
        color += attenuation * layer.rgb;
        attenuation *= layer.a;
        if (!hit) break;
        uint firstNormal = 3 * (triangleOffset(q.CommittedInstanceID()) + q.CommittedPrimitiveIndex());
        float2 bary = q.CommittedTriangleBarycentrics();
        float3 normal = normalize(triangleNormals[firstNormal].xyz * (1 - bary.x - bary.y)
            + triangleNormals[firstNormal + 1].xyz * bary.x + triangleNormals[firstNormal + 2].xyz * bary.y);
        if (dot(normal, direction) > 0) normal = -normal;
        attenuation *= 0.8;
        pathLength += t;
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
