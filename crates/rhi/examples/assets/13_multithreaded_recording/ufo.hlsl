cbuffer Transform : register(b0) { float4x4 mvp; float4 tint; };
struct Input { float3 position : LOCATION0; float3 normal : LOCATION1; float2 uv : LOCATION2; float3 color : LOCATION3; };
struct Output { float4 position : SV_Position; float3 normal : NORMAL0; float3 color : COLOR0; float3 view : TEXCOORD0; float3 light : TEXCOORD1; };
Output vs_main(Input input) {
    Output output;
    float4 position = mul(mvp, float4(input.position, 1));
    output.position = position;
    // Matches the original multithreading/phong.vert, including its clip-space
    // normal transform, which gives this demo its characteristic highlights.
    output.normal = mul((float3x3)mvp, input.normal);
    output.color = all(input.color == float3(1, 0, 0)) ? tint.rgb : input.color;
    output.light = -position.xyz;
    output.view = -position.xyz;
    return output;
}
float4 ps_main(Output input) : SV_TARGET {
    float3 normal = normalize(input.normal);
    float3 light = normalize(input.light);
    float3 view = normalize(input.view);
    float3 reflected = reflect(-light, normal);
    float3 diffuse = max(dot(normal, light), 0) * input.color;
    float3 specular = pow(max(dot(reflected, view), 0), 8) * float3(.75, .75, .75);
    return float4(diffuse + specular, 1);
}
