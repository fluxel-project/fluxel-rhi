cbuffer Params : register(b1) { uint pad0; uint pad1; uint a; uint b; };
RWByteAddressBuffer output : register(u2);
[numthreads(1,1,1)] void main() { output.Store2(0, uint2(a,b)); }
