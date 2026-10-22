StructuredBuffer<uint> input : register(t0);
RWStructuredBuffer<uint> output : register(u1);
[numthreads(1,1,1)] void main() { uint n, stride, m, out_stride; input.GetDimensions(n, stride); output.GetDimensions(m, out_stride); output[0]=n; output[1]=input[0]; output[2]=m; }
