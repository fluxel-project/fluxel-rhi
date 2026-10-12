// dxc -T vs_6_0 -E vs_main descriptor_indexing.hlsl -Fo descriptor_indexing.vs.dxil
// dxc -T ps_6_0 -E ps_main descriptor_indexing.hlsl -Fo descriptor_indexing.ps.dxil
cbuffer Uniforms : register(b0) { float4x4 projection; float4x4 view; float4x4 model; };
SamplerState nearest_sampler : register(s1);
Texture2D textures[] : register(t2);
struct Vertex { float3 position : LOCATION0; float2 uv : LOCATION1; int texture_index : LOCATION2; };
struct Varyings { float4 position : SV_POSITION; float2 uv : TEXCOORD0; nointerpolation int texture_index : TEXCOORD1; };
Varyings vs_main(Vertex i) { Varyings o; o.position=mul(projection,mul(view,mul(model,float4(i.position,1)))); o.uv=i.uv; o.texture_index=i.texture_index; return o; }
float4 ps_main(Varyings i) : SV_TARGET { return textures[NonUniformResourceIndex(uint(i.texture_index))].Sample(nearest_sampler,i.uv); }
