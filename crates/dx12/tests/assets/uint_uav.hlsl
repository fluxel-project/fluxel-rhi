RWTexture2DArray<uint> target : register(u0);
[numthreads(1, 1, 1)] void main() { target[uint3(0, 0, 0)] = 0x12345678; }
