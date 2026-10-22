// Controlled visual subset of NVIDIA Donut rt_particles; see README for differences.
// Copyright (c) 2014-2023 NVIDIA CORPORATION. MIT license, see assets directory.
RaytracingAccelerationStructure scene : register(t0);
RWTexture2D<float4> outputImage : register(u1);
StructuredBuffer<float4> triangleNormals : register(t2);
struct Particle { float4 centerRadius; float4 colorOpacity; float4 rotation; };
StructuredBuffer<Particle> particles : register(t3);
StructuredBuffer<uint> environmentPixels : register(t5);
StructuredBuffer<uint> smokePixels : register(t6);
StructuredBuffer<uint> uiPixels : register(t7);
cbuffer Frame : register(b4) {
    float4 cameraPosition;
    float4 cameraForward;
    float4 cameraRight;
    float4 cameraUp;
    uint4 dimensions; // width, height, particle count, feature flags
    uint4 triangleOffsets0;
    uint4 triangleOffsets1;
};
uint triangleOffset(uint instance) {
    return instance < 4 ? triangleOffsets0[instance] : triangleOffsets1[instance - 4];
}
// The source uses a wrapping linear sampler. The smoke buffer contains a mip chain;
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
    uint baseSize = (dimensions.w & 128) != 0 ? 768 : 512;
    int size = max(1u, baseSize >> mip);
    xy = (xy % size + size) % size;
    uint offset = (dimensions.w & 128) != 0 ? 349525 : 0;
    for (uint level = 0; level < mip; ++level) {
        uint levelSize = max(1u, baseSize >> level);
        offset += levelSize * levelSize;
    }
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
    uint baseSize = (dimensions.w & 128) != 0 ? 768 : 512;
    float size = float(max(1u, baseSize >> mip));
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
float4 particleLayer(float3 origin, float3 direction, float maxT, float pathLength, bool secondary,
                     float3x3 accumulatedTransform) {
    bool procedural = secondary ? (dimensions.w & 2) == 0 : (dimensions.w & 1) != 0;
    uint orientation = (dimensions.w >> 2) & 3;
    uint mode = (dimensions.w >> 4) & 3;
    uint layerCount = mode == 0 ? 4 : (mode == 1 ? 1 : (mode == 2 ? 2 : 8));
    // Primary quads and secondary procedural bounds have separate instance masks.
    // The reference defaults to four MLAB fragments. Each entry stores
    // premultiplied color, transmittance, and ray distance.
    float4 layers[8];
    float depths[8];
    for (uint j = 0; j < layerCount; ++j) {
        layers[j] = float4(0, 0, 0, 1);
        depths[j] = 1e30;
    }
    RayDesc particleRay;
    particleRay.Origin = origin;
    particleRay.Direction = direction;
    particleRay.TMin = 0;
    particleRay.TMax = maxT;
    RayQuery<RAY_FLAG_NONE> particleQuery;
    particleQuery.TraceRayInline(scene, RAY_FLAG_NONE, procedural ? 4 : 2, particleRay);
    while (particleQuery.Proceed()) {
        uint particleIndex;
        float t;
        float2 uv;
        if (procedural) {
            if (particleQuery.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) continue;
            particleIndex = particleQuery.CandidateInstanceID();
            if (particleIndex >= dimensions.z) continue;
            Particle p = particles[particleIndex];
            float3 right = cos(p.rotation.x) * cameraRight.xyz + sin(p.rotation.x) * cameraUp.xyz;
            float3 up = -sin(p.rotation.x) * cameraRight.xyz + cos(p.rotation.x) * cameraUp.xyz;
            if ((dimensions.w & 256) != 0) { right = cameraRight.xyz; up = float3(0, 1, 0); }
            float3 normal = normalize(cross(right, up));
            float3 virtualCamera = origin - direction * pathLength;
            float3 facingNormal = normalize(virtualCamera - p.centerRadius.xyz);
            if (orientation == 3) {
                right = mul(right, accumulatedTransform);
                up = mul(up, accumulatedTransform);
                normal = mul(normal, accumulatedTransform);
            } else if (orientation == 2) {
                float3 beamRight = cross(up, facingNormal);
                float beamLength = length(beamRight);
                if (beamLength > 0) right = beamRight / beamLength;
                normal = cross(right, up);
            } else if (orientation == 1) {
                normal = facingNormal;
                float3 basisUp = float3(0, 1, 0);
                if (abs(dot(basisUp, normal)) >= 0.999) basisUp = float3(1, 0, 0);
                float3 basisRight = normalize(cross(basisUp, normal));
                basisUp = normalize(cross(normal, basisRight));
                right = cos(p.rotation.x) * basisRight + sin(p.rotation.x) * basisUp;
                up = -sin(p.rotation.x) * basisRight + cos(p.rotation.x) * basisUp;
            } else {
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
            t = dot(p.centerRadius.xyz - origin, normal) / denominator;
            if (t <= 0 || t >= maxT) continue;
            float3 delta = origin + direction * t - p.centerRadius.xyz;
            uv = float2(dot(delta, right), -dot(delta, up)) / p.centerRadius.w * 0.5 + 0.5;
            if (any(uv < 0) || any(uv >= 1)) continue;
        } else {
            if (particleQuery.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) continue;
            uint primitive = particleQuery.CandidatePrimitiveIndex();
            particleIndex = primitive / 2;
            if (particleIndex >= dimensions.z) continue;
            t = particleQuery.CandidateTriangleRayT();
            float2 bary = particleQuery.CandidateTriangleBarycentrics();
            // Quad vertices are (top-left, top-right, bottom-right) and
            // (top-left, bottom-right, bottom-left), respectively.
            uv = (primitive & 1) == 0 ? float2(bary.x + bary.y, bary.y)
                                      : float2(bary.x, bary.x + bary.y);
        }
        Particle p = particles[particleIndex];
        float footprint = (pathLength + t) * (0.7853981634 / float(dimensions.y));
        float textureSize = (dimensions.w & 128) != 0 ? 768.0 : 512.0;
        float mip = log2(0.5 * textureSize * footprint / p.centerRadius.w);
        float4 smoke = sampleSmoke(uv, mip);
        float3 texel = smoke.rgb;
        float alpha = smoke.a * p.colorOpacity.a;
        if (alpha == 0) continue;
        float4 fragment = float4(alpha * p.colorOpacity.rgb * texel, 1.0 - alpha);
        float depth = t;
        for (uint j = 0; j < layerCount; ++j) {
            if (depth < depths[j]) {
                float4 previous = layers[j];
                float previousDepth = depths[j];
                layers[j] = fragment;
                depths[j] = depth;
                fragment = previous;
                depth = previousDepth;
            }
        }
        layers[layerCount - 1].rgb += fragment.rgb * layers[layerCount - 1].a;
        layers[layerCount - 1].a *= fragment.a;
    }
    float4 result = layers[0];
    for (uint j = 1; j < layerCount; ++j) {
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
    float3x3 accumulatedTransform = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    for (uint bounce = 0; bounce < 8; ++bounce) {
        RayDesc ray; ray.Origin = origin; ray.Direction = direction; ray.TMin = 0; ray.TMax = 1000;
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES> q;
        q.TraceRayInline(scene, RAY_FLAG_NONE, 1, ray);
        while (q.Proceed()) { }
        bool hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
        float t = hit ? q.CommittedRayT() : ray.TMax;
        float4 layer = particleLayer(origin, direction, t, pathLength, bounce > 0, accumulatedTransform);
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
        float3x3 reflection = float3x3(
            1 - 2 * normal.x * normal.x, -2 * normal.x * normal.y, -2 * normal.x * normal.z,
            -2 * normal.y * normal.x, 1 - 2 * normal.y * normal.y, -2 * normal.y * normal.z,
            -2 * normal.z * normal.x, -2 * normal.z * normal.y, 1 - 2 * normal.z * normal.z);
        accumulatedTransform = mul(accumulatedTransform, reflection);
    }
    color += attenuation * environmentColor(direction);
    // The reference blits its linear RGBA16_FLOAT buffer to an sRGB swapchain.
    color = saturate(color);
    float3 srgb = lerp(12.92 * color, 1.055 * pow(color, 1.0 / 2.4) - 0.055,
                       step(0.0031308, color));
    if ((dimensions.w & 64) != 0 && pixel.x >= 15 && pixel.y >= 15 &&
        pixel.x < 390 && pixel.y < 465) {
        uint2 uiCoord = (pixel - uint2(15, 15)) * uint2(320, 310) / uint2(375, 450);
        float4 ui = unpackColor(uiPixels[uiCoord.y * 320 + uiCoord.x]);
        srgb = lerp(srgb, ui.rgb, ui.a);
    }
    outputImage[pixel] = float4(srgb, 1);
}
