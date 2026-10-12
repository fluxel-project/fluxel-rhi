// Matches shaders/glsl/multiview/multiview.{vert,frag}: the light and
// normals are transformed independently for each native multiview index.
cbuffer Eyes : register(b0) {
    float4x4 projection[2];
    float4x4 modelview[2];
    float4 light_pos;
};
struct In { float3 position : LOCATION0; float3 normal : LOCATION1; float2 uv : LOCATION2; float3 color : LOCATION3; };
struct Out {
    float4 position : SV_Position;
    float3 normal : NORMAL0;
    float3 color : COLOR0;
    float3 view_vec : TEXCOORD0;
    float3 light_vec : TEXCOORD1;
};
Out vs_main(In input, uint view : SV_ViewID) {
    Out output;
    float4 world_pos = mul(modelview[view], float4(input.position, 1.0));
    output.position = mul(projection[view], world_pos);
    output.normal = mul((float3x3)modelview[view], input.normal);
    output.color = input.color;
    output.light_vec = mul(modelview[view], light_pos).xyz - world_pos.xyz;
    output.view_vec = -world_pos.xyz;
    return output;
}
float4 ps_main(Out input) : SV_Target {
    float3 n = normalize(input.normal);
    float3 l = normalize(input.light_vec);
    float3 v = normalize(input.view_vec);
    float3 r = reflect(-l, n);
    float3 ambient = float3(0.1, 0.1, 0.1);
    float3 diffuse = max(dot(n, l), 0.0) * float3(1.0, 1.0, 1.0);
    float3 specular = pow(max(dot(r, v), 0.0), 16.0) * float3(0.75, 0.75, 0.75);
    return float4((ambient + diffuse) * input.color + specular, 1.0);
}
