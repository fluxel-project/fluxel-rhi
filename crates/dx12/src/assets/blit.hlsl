// Internal rectangle blit. Constants describe the source rectangle in normalized coordinates.
cbuffer BlitConstants : register(b0) { float2 uv_origin; float2 uv_size; float source_z; };
#ifdef BLIT_UINT
#define BLIT_TYPE uint4
#elif defined(BLIT_SINT)
#define BLIT_TYPE int4
#else
#define BLIT_TYPE float4
#endif
#if defined(BLIT_D1)
Texture1DArray<BLIT_TYPE> image : register(t0);
#elif defined(BLIT_D3)
Texture3D<BLIT_TYPE> image : register(t0);
#elif defined(BLIT_ARRAY)
Texture2DArray<BLIT_TYPE> image : register(t0);
#else
Texture2D<BLIT_TYPE> image : register(t0);
#endif
SamplerState image_sampler : register(s0);
struct Varyings { float4 position : SV_Position; float2 uv : TEXCOORD0; };
Varyings vs(uint id : SV_VertexID) { Varyings o; float2 p = float2((id << 1) & 2, id & 2); o.position = float4(p * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = p; return o; }
BLIT_TYPE ps(Varyings i) : SV_Target {
#if defined(BLIT_UINT) || defined(BLIT_SINT)
#if defined(BLIT_D1)
    uint width, layers;
    image.GetDimensions(width, layers);
    int x = int(clamp((uv_origin.x + i.uv.x * uv_size.x) * width, 0, width - 1));
    return image.Load(int3(x, 0, 0));
#elif defined(BLIT_D3)
    uint width, height, depth;
    image.GetDimensions(width, height, depth);
    int3 pixel = int3(clamp(float3(uv_origin + i.uv * uv_size, source_z) * float3(width, height, depth), 0, float3(width - 1, height - 1, depth - 1)));
    return image.Load(int4(pixel, 0));
#else
    uint width, height;
#ifdef BLIT_ARRAY
    uint layers;
    image.GetDimensions(width, height, layers);
#else
    image.GetDimensions(width, height);
#endif
    int2 pixel = int2(clamp((uv_origin + i.uv * uv_size) * float2(width, height), float2(0, 0), float2(width - 1, height - 1)));
#ifdef BLIT_ARRAY
    return image.Load(int4(pixel, 0, 0));
#else
    return image.Load(int3(pixel, 0));
#endif
#endif
#else
#if defined(BLIT_D1)
    return image.SampleLevel(image_sampler, float2(uv_origin.x + i.uv.x * uv_size.x, 0), 0);
#elif defined(BLIT_D3)
    return image.SampleLevel(image_sampler, float3(uv_origin + i.uv * uv_size, source_z), 0);
#elif defined(BLIT_ARRAY)
    return image.SampleLevel(image_sampler, float3(uv_origin + i.uv * uv_size, 0), 0);
#else
    return image.SampleLevel(image_sampler, uv_origin + i.uv * uv_size, 0);
#endif
#endif
}

#if defined(BLIT_COMPUTE)
#if defined(DEST_D1)
RWTexture1D<BLIT_TYPE> output_image : register(u0);
#elif defined(DEST_D3)
RWTexture3D<BLIT_TYPE> output_image : register(u0);
#else
RWTexture2D<BLIT_TYPE> output_image : register(u0);
#endif
cbuffer OutputConstants : register(b1) { uint2 output_size; uint encode_srgb; };
[numthreads(8,8,1)]
void cs(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= output_size)) return;
    Varyings v; v.position = 0; v.uv = (float2(id.xy) + 0.5) / float2(output_size);
    BLIT_TYPE color = ps(v);
#if !defined(BLIT_UINT) && !defined(BLIT_SINT)
    if (encode_srgb != 0) color.rgb = lerp(12.92 * color.rgb, 1.055 * pow(max(color.rgb, 0), 1.0/2.4) - 0.055, step(0.0031308, color.rgb));
#endif
#if defined(DEST_D1)
    output_image[id.x] = color;
#elif defined(DEST_D3)
    output_image[uint3(id.xy, 0)] = color;
#else
    output_image[id.xy] = color;
#endif
}
#endif
