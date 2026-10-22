cbuffer Params : register(b1) { uint x; uint unused; uint y; uint pad; };
float4 main(float4 pos:SV_Position, float4 col:COLOR0):SV_Target { return y == 30 ? col : float4(0,0,1,1); }
