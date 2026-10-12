cbuffer Eyes : register(b0) { float4x4 mvp[2]; };
struct In { float3 position : LOCATION0; float3 normal : LOCATION1; float2 uv : LOCATION2; float3 color : LOCATION3; };
struct Out { float4 position : SV_Position; float3 color : COLOR0; };
Out vs_main(In input, uint view : SV_ViewID) {
    Out output;
    output.position = mul(mvp[view], float4(input.position, 1.0));
    float lit = max(dot(normalize(input.normal), normalize(float3(-2, 4, -3))), 0.2);
    output.color = input.color * lit;
    return output;
}
float4 ps_main(Out input) : SV_Target { return float4(input.color, 1.0); }