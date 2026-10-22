RWByteAddressBuffer result : register(u0);
RWByteAddressBuffer unused_values[2] : register(u2);
[numthreads(1,1,1)] void main() { result.Store(0, 123); }
