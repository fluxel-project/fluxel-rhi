ByteAddressBuffer source : register(t0);
RWByteAddressBuffer result : register(u1);
cbuffer Params : register(b2) { uint value; };
[shader("raygeneration")]
void raygen() { result.Store(0, source.Load(0)); result.Store(4, value); }
