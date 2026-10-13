// Native DXIL/SPIR-V triangle path. Attribute locations match example 01.
struct VertexInput {
    float3 position : LOCATION0;
    float3 color : LOCATION1;
};

struct VertexOutput {
    float4 position : SV_POSITION;
    float3 color : COLOR0;
};

VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 1.0);
    output.color = input.color;
    return output;
}

float4 ps_main(VertexOutput input) : SV_TARGET {
    return float4(input.color, 1.0);
}
