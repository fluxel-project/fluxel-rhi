cbuffer Params : register(b0) { uint value; }
RWByteAddressBuffer output : register(u1);
[numthreads(1,1,1)] void main(uint3 id : SV_DispatchThreadID) { output.Store(id.x * 4, value + id.x); }
