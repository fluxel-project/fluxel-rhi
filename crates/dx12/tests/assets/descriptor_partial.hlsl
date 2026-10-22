RWByteAddressBuffer values[2] : register(u0);
[numthreads(1,1,1)] void main() { values[1].Store(0, 99); }
