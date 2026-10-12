// DX12 API ABI: b0 is a CBV and b1 is eight root constants (32 bytes).
// Keep the source sample's operation order: scale the mesh in model space,
// then add the per-sphere push-constant position in world space.
cbuffer Camera : register(b0, space0) {
    float4x4 projection_view;
    float4x4 model;
}
cbuffer Push : register(b1, space0) { float4 color; float4 offset; }
struct VSIn { float3 position: LOCATION0; float3 normal:LOCATION1; float4 base:LOCATION2; };
struct VSOut { float4 position:SV_Position;float3 color:COLOR0; };
VSOut vs_main(VSIn i){
    VSOut o;
    float3 local = mul(model, float4(i.position, 1)).xyz;
    o.position=mul(projection_view,float4(local + offset.xyz,1));
    o.color=i.base.rgb*color.rgb;
    return o;
}
float4 fs_main(VSOut i):SV_Target{return float4(i.color,1);}
