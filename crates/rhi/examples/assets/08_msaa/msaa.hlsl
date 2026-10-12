cbuffer Transform : register(b0) {
    float4x4 projection;
    float4x4 model;
    float4 light_position;
};
Texture2D base_color : register(t1);
SamplerState base_sampler : register(s2);
struct VertexInput {
    float3 position : LOCATION0;
    float3 normal : LOCATION1;
    float2 uv : LOCATION2;
    float3 color : LOCATION3;
};
struct VertexOutput {
    float4 position : SV_Position;
    float3 normal : TEXCOORD0;
    float2 uv : TEXCOORD1;
    float3 color : COLOR0;
};
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    float4 world = mul(model, float4(input.position, 1.0));
    output.position = mul(projection, world);
    output.normal = normalize(mul((float3x3)model, input.normal));
    output.uv = input.uv;
    output.color = input.color;
    return output;
}
float4 ps_main(VertexOutput input) : SV_Target {
    float3 light_dir = normalize(light_position.xyz - input.position.xyz);
    float diffuse = max(dot(normalize(input.normal), light_dir), 0.12);
    return float4(base_color.Sample(base_sampler, input.uv).rgb * input.color * diffuse, 1.0);
}
