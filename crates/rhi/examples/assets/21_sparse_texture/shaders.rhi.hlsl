// Single-mip RHI ABI adaptation of Microsoft's ReservedResources shader.
struct PSInput { float4 position : SV_POSITION; float2 uv : TEXCOORD; };
Texture2D g_texture : register(t1);
SamplerState g_sampler : register(s2);
PSInput VSMain(float4 position : LOCATION0, float2 uv : LOCATION1) {
    PSInput result; result.position = position; result.uv = uv; return result;
}
float4 PSMain(PSInput input) : SV_TARGET { return g_texture.SampleLevel(g_sampler, input.uv, 0); }
