ByteAddressBuffer data : register(t0);
cbuffer Params : register(b2) { uint expected; };
struct V { float4 pos:SV_Position; float4 col:COLOR0; };
V main(uint id:SV_VertexID) { V o; float2 p = id==0 ? float2(-1,-1) : id==1 ? float2(-1,3) : float2(3,-1); o.pos=float4(p,0,1); o.col = data.Load(0)==expected ? float4(1,0,0,1) : float4(0,1,0,1); return o; }
