// Native dynamic-uniform ABI: camera at b0 and the selected model matrix at b1.
cbuffer Camera : register(b0) { float4x4 vp; }
cbuffer Model : register(b1) { float4x4 model; }
struct VSIn { float3 position: LOCATION0; float3 color: LOCATION1; };
struct VSOut { float4 position: SV_Position; float3 color: COLOR0; };
VSOut vs_main(VSIn i) { VSOut o; o.position=mul(vp,mul(model,float4(i.position,1)));o.color=i.color;return o; }
float4 fs_main(VSOut i):SV_Target{return float4(i.color,1);}
