// Typed-format fallback: one thread owns a DWORD, including sub-DWORD pixels.
#include "blit.hlsl"
RWByteAddressBuffer output_buffer : register(u0);
cbuffer OutputConstants : register(b1) { uint2 output_size; uint srgb; uint row_pitch; uint format_code; uint pixel_bytes; };
uint pack_ufloat(float value, uint mantissa_bits) {
    uint h = f32tof16(max(value, 0));
    uint shift = 10 - mantissa_bits;
    uint rounded = h + ((1u << (shift-1)) - 1) + ((h >> shift) & 1);
    return min(rounded >> shift, 31u << mantissa_bits);
}
uint4 encode_pixel(uint x, uint y) {
    Varyings v; v.position = 0; v.uv = (float2(x,y)+0.5)/float2(output_size);
    BLIT_TYPE c = ps(v);
#if !defined(BLIT_UINT) && !defined(BLIT_SINT)
    if (srgb) c.rgb = lerp(12.92*c.rgb, 1.055*pow(max(c.rgb,0),1.0/2.4)-0.055, step(0.0031308,c.rgb));
#endif
    uint bits = format_code & 255, count = (format_code >> 8) & 255, type = (format_code >> 16) & 255;
    uint4 a = 0;
#if !defined(BLIT_UINT) && !defined(BLIT_SINT)
    if (type == 5) return uint4(pack_ufloat(c.r,6) | (pack_ufloat(c.g,6)<<11) | (pack_ufloat(c.b,5)<<22),0,0,0);
    if (type == 6) {
        float3 rgb = clamp(c.rgb, 0, 65408);
        float maxc = max(rgb.r, max(rgb.g,rgb.b));
        if (maxc < exp2(-25.0)) return 0;
        int e = clamp(int(floor(log2(maxc))) + 16, 0, 31);
        float scale = exp2(float(e-24));
        if (round(maxc/scale) >= 512 && e < 31) { ++e; scale *= 2; }
        uint3 q = min(uint3(round(rgb/scale)),511);
        return uint4(q.x | (q.y<<9) | (q.z<<18) | (uint(e)<<27),0,0,0);
    }
#endif
    if (type == 4) {
#if defined(BLIT_UINT)
        uint4 q = min(c, uint4(1023,1023,1023,3));
#else
        uint4 q = uint4(round(saturate(c)*float4(1023,1023,1023,3)));
#endif
        return uint4(q.x | (q.y<<10) | (q.z<<20) | (q.w<<30),0,0,0);
    }
#if defined(BLIT_UINT)
    a = min(c, bits == 32 ? 0xffffffff : ((1u << bits) - 1));
#elif defined(BLIT_SINT)
    int limit = bits == 32 ? 0x7fffffff : int((1u << (bits-1)) - 1);
    a = asuint(clamp(c, -limit - 1, limit));
#else
    if (type == 0) a = uint4(round(saturate(c)*((1u << bits)-1)));
    else if (type == 1) { int limit = int((1u << (bits-1))-1); a = asuint(int4(round(clamp(c,-1,1)*limit))); }
    else if (bits == 16) a = f32tof16(c);
    else a = asuint(c);
#endif
    if (bits < 32) a &= (1u << bits)-1;
    if ((format_code & 0x1000000) != 0) a = a.bgra;
    if (bits == 8) return uint4(a.x | (a.y<<8) | (a.z<<16) | (a.w<<24),0,0,0);
    if (bits == 16) return uint4(a.x | (a.y<<16), a.z | (a.w<<16), 0,0);
    return a;
}
[numthreads(8,8,1)] void cs_buffer(uint3 id : SV_DispatchThreadID) {
    uint pixels_per_word = max(1u, 4u / pixel_bytes);
    uint x = id.x * pixels_per_word;
    if (x >= output_size.x || id.y >= output_size.y) return;
    uint address = id.y*row_pitch + x*pixel_bytes;
    uint4 packed = encode_pixel(x,id.y);
    if (pixel_bytes < 4) {
        uint word = packed.x & (pixel_bytes == 1 ? 255u : 65535u);
        for (uint n=1; n<pixels_per_word && x+n<output_size.x; ++n)
            word |= (encode_pixel(x+n,id.y).x & (pixel_bytes == 1 ? 255u : 65535u)) << (8*pixel_bytes*n);
        output_buffer.Store(address,word);
    } else {
        output_buffer.Store(address,packed.x);
        if (pixel_bytes >= 8) output_buffer.Store(address+4,packed.y);
        if (pixel_bytes >= 16) { output_buffer.Store(address+8,packed.z); output_buffer.Store(address+12,packed.w); }
    }
}
