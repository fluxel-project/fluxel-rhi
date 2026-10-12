Texture2D<float4> source_image : register(t0);
#ifdef FLUXEL_VULKAN
[[vk::image_format("rgba8")]]
#endif
RWTexture2D<float4> result_image : register(u1);

float gray(int2 p) { float3 c = source_image.Load(int3(p, 0)).rgb; return dot(c, float3(.33333334, .33333334, .33333334)); }
float convolution(int2 p, float k[9], float divisor, float bias) {
    float sum = 0; uint n = 0;
    [unroll] for (int y = -1; y <= 1; ++y) [unroll] for (int x = -1; x <= 1; ++x) sum += gray(p + int2(x,y)) * k[n++];
    return saturate(sum / divisor + bias);
}
[numthreads(16,16,1)] void emboss(uint3 id:SV_DispatchThreadID) { float k[9]={-1,0,0,0,-1,0,0,0,2}; float v=convolution(id.xy,k,1,.5); result_image[id.xy]=float4(v,v,v,1); }
[numthreads(16,16,1)] void edgedetect(uint3 id:SV_DispatchThreadID) { float k[9]={-.125,-.125,-.125,-.125,1,-.125,-.125,-.125,-.125}; float v=convolution(id.xy,k,.1,0); result_image[id.xy]=float4(v,v,v,1); }
[numthreads(16,16,1)] void sharpen(uint3 id:SV_DispatchThreadID) { float3 s=0; uint n=0; float k[9]={-1,-1,-1,-1,9,-1,-1,-1,-1}; [unroll] for(int y=-1;y<=1;++y)[unroll]for(int x=-1;x<=1;++x)s+=source_image.Load(int3(int2(id.xy)+int2(x,y),0)).rgb*k[n++]; result_image[id.xy]=float4(saturate(s),1); }
