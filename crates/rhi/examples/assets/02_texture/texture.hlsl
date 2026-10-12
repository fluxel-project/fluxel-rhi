cbuffer Transform : register(b0) { float4x4 transform; }
Texture2D color_texture : register(t1); SamplerState color_sampler : register(s2);
struct VSIn { float3 position: LOCATION0; float2 uv: LOCATION1; };
struct VSOut { float4 position: SV_Position; float2 uv: TEXCOORD0; };
VSOut vs_main(VSIn i) { VSOut o; o.position=mul(transform,float4(i.position,1)); o.uv=i.uv; return o; }
float4 fs_main(VSOut i) : SV_Target { return color_texture.Sample(color_sampler,i.uv); }
