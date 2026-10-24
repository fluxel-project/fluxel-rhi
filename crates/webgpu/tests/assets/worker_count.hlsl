RWByteAddressBuffer data : register(u0);
[numthreads(1,1,1)] void main() { data.Store(64,0); data.Store(68,1); data.Store(72,2); data.Store(76,9); }
