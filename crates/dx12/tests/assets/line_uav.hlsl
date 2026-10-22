RWTexture1D<float4> target : register(u0);
[numthreads(1, 1, 1)] void main() { target[0] = float4(1, 0, 0, 1); }
