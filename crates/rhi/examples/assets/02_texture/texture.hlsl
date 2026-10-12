// ABI-equivalent to Vulkan's texture example: projection, camera view and
// view position occupy the uniform block; the shader keeps its diffuse and
// specular lighting instead of displaying an unlit texture.
cbuffer Scene : register(b0) {
    float4x4 projection;
    float4x4 model_view;
    float4 view_pos;
    float lod_bias;
    float3 _padding;
}
Texture2D color_texture : register(t1);
SamplerState color_sampler : register(s2);

struct VSIn {
    float3 position : LOCATION0;
    float2 uv : LOCATION1;
    float3 normal : LOCATION2;
};
struct VSOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float lod_bias : TEXCOORD1;
    float3 normal : NORMAL0;
    float3 view_vec : TEXCOORD2;
    float3 light_vec : TEXCOORD3;
};

VSOut vs_main(VSIn i) {
    VSOut o;
    float4 position = mul(model_view, float4(i.position, 1.0));
    o.position = mul(projection, position);
    o.uv = i.uv;
    o.lod_bias = lod_bias;
    o.normal = mul((float3x3)model_view, i.normal);
    // The source places the light at the model-view origin.
    o.light_vec = -position.xyz;
    o.view_vec = view_pos.xyz - position.xyz;
    return o;
}

float4 fs_main(VSOut i) : SV_Target {
    float4 color = color_texture.SampleLevel(color_sampler, i.uv, i.lod_bias);
    float3 normal = normalize(i.normal);
    float3 light = normalize(i.light_vec);
    float3 view = normalize(i.view_vec);
    float3 reflection = reflect(-light, normal);
    float3 diffuse = max(dot(normal, light), 0.0) * float3(1.0, 1.0, 1.0);
    float specular = pow(max(dot(reflection, view), 0.0), 16.0) * color.a;
    return float4(diffuse * color.rgb + specular, 1.0);
}
