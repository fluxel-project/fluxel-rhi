RWByteAddressBuffer args : register(u0);
[numthreads(1,1,1)] void main() { args.Store3(4, uint3(3, 1, 1)); }
