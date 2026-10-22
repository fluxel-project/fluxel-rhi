cbuffer Params : register(b1) { uint x; uint unused; uint y; uint pad; };
struct V { float4 pos:SV_Position; float4 col:COLOR0; };
V main(uint id:SV_VertexID) { V o; float2 p = id==0 ? float2(-1,-1) : id==1 ? float2(-1,3) : float2(3,-1); o.pos=float4(p,0,1); o.col = x == 12 ? float4(1,0,0,1) : float4(0,1,0,1); return o; }
