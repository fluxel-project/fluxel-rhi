cbuffer Transform : register(b0) { float4x4 mvp; float4 tint; };
struct I { float3 p : LOCATION0; float3 n : LOCATION1; float2 uv : LOCATION2; float3 c : LOCATION3; };
struct P { float4 p : SV_Position; float3 n : NORMAL0; float3 c : COLOR0; float3 v : TEXCOORD0; };
P vs_main(I i) { P o; o.p = mul(mvp, float4(i.p, 1)); o.n = normalize(i.n); o.c = tint.rgb; o.v = -i.p; return o; }
float4 ps_main(P i) : SV_Target { float3 n=normalize(i.n); float3 l=normalize(float3(-2,4,-3)); float3 v=normalize(i.v); float3 r=reflect(-l,n); float d=max(dot(n,l),.15); float s=pow(max(dot(r,v),0),16)*.5; return float4(i.c*(.25+d)+s, tint.a); }