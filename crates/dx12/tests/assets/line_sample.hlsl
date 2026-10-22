Texture1D<float4> image : register(t0);
RWByteAddressBuffer result : register(u1);
[numthreads(1,1,1)] void main() { result.Store4(0, asuint(image.Load(int2(1,0)))); }
