struct VsIn { float3 position : LOCATION0; float3 color : LOCATION1; };
struct VsOut { float4 position : SV_Position; float3 color : COLOR0; };
VsOut vs_main(VsIn input) { VsOut o; o.position=float4(input.position,1.0); o.color=input.color; return o; }
float4 ps_main(VsOut input) : SV_Target { return float4(input.color,1.0); }
