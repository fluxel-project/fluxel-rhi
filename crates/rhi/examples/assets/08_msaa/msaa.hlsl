// Native Voyager material shader; its output is resolved by the RHI pass setup.
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
    float3 view_vector : TEXCOORD2;
    float3 light_vector : TEXCOORD3;
};
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    float4 world = mul(model, float4(input.position, 1.0));
    output.position = mul(projection, world);
    output.normal = normalize(mul((float3x3)model, input.normal));
    output.uv = input.uv;
    output.color = input.color;
    // The Vulkan source lights in view space.  SV_Position is in framebuffer
    // pixels at this stage, so it cannot be used as a lighting position.
    float3 light = mul((float3x3)model, light_position.xyz);
    output.light_vector = light - world.xyz;
    output.view_vector = -world.xyz;
    return output;
}
float4 ps_main(VertexOutput input) : SV_Target {
    float3 normal = normalize(input.normal);
    float3 light_dir = normalize(input.light_vector);
    float3 view_dir = normalize(input.view_vector);
    float3 reflected = reflect(-light_dir, normal);
    float3 color = base_color.Sample(base_sampler, input.uv).rgb * input.color;
    float diffuse = max(dot(normal, light_dir), 0.15);
    float specular = pow(max(dot(reflected, view_dir), 0.0), 16.0) * 0.75;
    return float4(color * diffuse + specular.xxx, 1.0);
}
