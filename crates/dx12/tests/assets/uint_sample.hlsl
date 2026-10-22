Texture2D<uint4> image : register(t0);
RWByteAddressBuffer result : register(u1);
[numthreads(1,1,1)] void main() { result.Store4(0, asuint(float4(image.Load(int3(1,1,0))))); }
