TextureCube<float4> image : register(t0);
RWByteAddressBuffer result : register(u1);
SamplerState sampler0 : register(s2);
[numthreads(1,1,1)] void main() { result.Store4(0, asuint(image.SampleLevel(sampler0, float3(1,0,0), 0))); }
