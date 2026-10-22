ByteAddressBuffer input_data : register(t0);
RWByteAddressBuffer output_data : register(u1);
cbuffer Params : register(b1) { uint value; };
[numthreads(1,1,1)] void main() { uint a = input_data.Load(0); output_data.Store(0, a); output_data.Store(4, value); output_data.Store(8, a + value); }
