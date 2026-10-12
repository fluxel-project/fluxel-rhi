// Matches the three GLSL pipelines in Vulkan/examples/occlusionquery.  The
// same vertex ABI is used by all of the embedded vkglTF models.
cbuffer Transform : register(b0) {
    float4x4 projection;
    float4x4 view;
    float4x4 model;
    float4 color;
    float4 light_pos;
    float visible;
    float3 _padding;
};
struct I { float3 p : LOCATION0; float3 n : LOCATION1; float2 uv : LOCATION2; float3 c : LOCATION3; };
struct P { float4 p : SV_Position; float3 n : NORMAL0; float3 c : COLOR0; float visible : TEXCOORD0; float3 view_vec : TEXCOORD1; float3 light_vec : TEXCOORD2; };
P vs_main(I i) {
    P o;
    float4 world = mul(model, float4(i.p, 1));
    o.p = mul(projection, mul(view, world));
    o.n = mul((float3x3)model, i.n);
    o.c = i.c * color.rgb;
    o.visible = visible;
    o.view_vec = -world.xyz;
    o.light_vec = light_pos.xyz - world.xyz;
    return o;
}
float4 simple_ps(P i) : SV_Target { return float4(1, 1, 1, 1); }
float4 mesh_ps(P i) : SV_Target {
    if (i.visible <= 0) return float4(.1, .1, .1, 1);
    float3 n = normalize(i.n), l = normalize(i.light_vec), v = normalize(i.view_vec);
    float3 diffuse = max(dot(n, l), .25) * i.c;
    float3 specular = pow(max(dot(reflect(-l, n), v), 0), 8) * .75;
    return float4(diffuse + specular, 1);
}
float4 occluder_ps(P i) : SV_Target { return float4(i.c, .5); }
