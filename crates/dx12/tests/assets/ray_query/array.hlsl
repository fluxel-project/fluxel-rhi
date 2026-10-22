RaytracingAccelerationStructure scene[2] : register(t0);
RWTexture2D<float4> output_image : register(u1);
[numthreads(2,1,1)] void main(uint3 id : SV_DispatchThreadID) {
    RayDesc ray; ray.Origin = float3(0,0,-1); ray.Direction = float3(0,0,1); ray.TMin = 0; ray.TMax = 10;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(scene[NonUniformResourceIndex(id.x)], RAY_FLAG_NONE, 255, ray);
    while (query.Proceed()) { }
    output_image[id.xy] = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? float4(1,0,0,1) : float4(0,1,0,1);
}
