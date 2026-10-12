struct In { float3 position : LOCATION0; float3 color : LOCATION1; };
struct Out { float4 position : SV_Position; float3 color : COLOR0; };
Out vs_main(In input, uint view : SV_ViewID) {
    Out output;
    float eye = view == 0 ? -0.18 : 0.18;
    output.position = float4(input.position.xy + float2(eye, 0.0), input.position.z, 1.0);
    output.color = input.color * (view == 0 ? float3(0.55, 0.8, 1.0) : float3(1.0, 0.55, 0.35));
    return output;
}
float4 ps_main(Out input) : SV_Target { return float4(input.color, 1.0); }
