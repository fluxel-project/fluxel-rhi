Texture2D<float4> input_tex[2] : register(t0);
SamplerState samplers[2] : register(s1);
RWTexture2DArray<float4> output_tex[2] : register(u2);
[numthreads(1,1,1)] void main() {
    output_tex[0][uint3(0,0,1)] = input_tex[0].SampleLevel(samplers[0], float2(0.5,0.5), 0);
    output_tex[1][uint3(0,0,1)] = input_tex[1].SampleLevel(samplers[1], float2(0.5,0.5), 0);
}
