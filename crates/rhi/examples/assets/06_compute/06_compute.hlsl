RWTexture2D<float4> src : register(u0);
RWTexture2D<float4> dst : register(u1);

[numthreads(16, 16, 1)]
void filter(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    dst.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) return;
    float4 c = src[id.xy];
    float luma = dot(c.rgb, float3(0.299, 0.587, 0.114));
    dst[id.xy] = float4(luma, luma, luma, c.a);
}
