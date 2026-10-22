cbuffer Params : register(b1) { uint a; uint unused; uint b; uint pad; };
RWByteAddressBuffer output : register(u2);
[numthreads(1,1,1)] void main() { output.Store2(0, uint2(a,b)); }
