RaytracingAccelerationStructure scene : register(t0);
RWTexture2D<float4> output_image : register(u1);
[numthreads(2, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    RayDesc ray;
    ray.Origin = id.x == 0 ? float3(0, 0, -1) : float3(4, 0, -1);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.0;
    ray.TMax = 10.0;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 255, ray);
    while (query.Proceed()) { }
    output_image[id.xy] = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT
        ? float4(1, 0, 0, 1) : float4(0, 1, 0, 1);
}
