struct V { float4 pos:SV_Position; float4 col:COLOR0; };
V main(uint id:SV_VertexID) { V o; uint i=id%3; float2 p = i==0 ? float2(-0.9,-0.8) : i==1 ? float2(-0.5,0.8) : float2(-0.1,-0.8); if(id>=3) p.x+=1; o.pos=float4(p,0,1); o.col=id>=3 ? float4(0,1,0,1) : float4(1,0,0,1); return o; }
