cbuffer U0 : register(b0) { uint a; };
cbuffer U1 : register(b1) { uint b; };
RWByteAddressBuffer output : register(u2);
[numthreads(1,1,1)] void main() { output.Store(0, a + b); }
