Texture2D<float4> source_texture : register(t0);
SamplerState source_sampler : register(s0);

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

VertexOutput vs_main(uint id : SV_VertexID) {
    const float2 positions[3] = { float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0) };
    const float2 uvs[3] = { float2(0.0, 1.0), float2(0.0, -1.0), float2(2.0, 1.0) };
    VertexOutput output;
    output.position = float4(positions[id], 0.0, 1.0);
    output.uv = uvs[id];
    return output;
}

float4 ps_main(VertexOutput input) : SV_Target0 {
    return source_texture.SampleLevel(source_sampler, input.uv, 0.0);
}
