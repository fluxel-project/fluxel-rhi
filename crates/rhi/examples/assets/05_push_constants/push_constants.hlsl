// DX12 API ABI: b0 is a CBV and b1 is eight root constants (32 bytes).
// The backend supplies this root parameter from RasterExtras::push_constants.
cbuffer Camera : register(b0, space0) { float4x4 vp; }
cbuffer Push : register(b1, space0) { float4 color; float4 offset; }
struct VSIn { float3 position: LOCATION0; float3 normal:LOCATION1; float4 base:LOCATION2; };
struct VSOut { float4 position:SV_Position;float3 color:COLOR0; };
VSOut vs_main(VSIn i){VSOut o;o.position=mul(vp,float4(i.position+offset.xyz,1));o.color=i.base.rgb*color.rgb;return o;}
float4 fs_main(VSOut i):SV_Target{return float4(i.color,1);}
