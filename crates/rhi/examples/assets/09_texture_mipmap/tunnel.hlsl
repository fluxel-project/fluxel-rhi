// Native textured tunnel pass consuming the generated mip chain.
cbuffer Scene : register(b0) {
    column_major float4x4 projection;
    column_major float4x4 view;
    column_major float4x4 model;
    float4 view_pos;
    float lod_bias;
};

Texture2D albedo : register(t1);
SamplerState sampler0 : register(s2);

struct VertexInput {
    float3 position : LOCATION0;
    float2 uv : LOCATION1;
    float3 normal : LOCATION2;
};
struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float lod_bias : TEXCOORD1;
    float3 normal : NORMAL0;
    float3 view_vector : TEXCOORD2;
    float3 light_vector : TEXCOORD3;
};

VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    float4 world = mul(model, float4(input.position, 1.0));
    output.position = mul(projection, mul(view, world));
    output.uv = input.uv * float2(2.0, 1.0);
    output.lod_bias = lod_bias;
    output.normal = mul((float3x3)model, input.normal);
    output.light_vector = world.xyz - float3(-30.0, 0.0, 0.0);
    output.view_vector = view_pos.xyz - world.xyz;
    return output;
}

float4 ps_main(VertexOutput input) : SV_Target {
    float4 color = albedo.SampleBias(sampler0, input.uv, input.lod_bias);
    float3 normal = normalize(input.normal);
    float3 light = normalize(input.light_vector);
    float3 view = normalize(input.view_vector);
    float3 reflection = reflect(light, normal);
    float3 diffuse = max(dot(normal, light), 0.65).xxx;
    float specular = pow(max(dot(reflection, view), 0.0), 16.0) * color.a;
    return float4(diffuse * color.rgb + specular, 1.0);
}
