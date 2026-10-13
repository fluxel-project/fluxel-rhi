// Native display pass samples the two multiview layers into left and right halves.
Texture2DArray<float4> views : register(t0);
SamplerState view_sampler : register(s1);
struct Out { float4 position : SV_Position; float2 uv : TEXCOORD0; };
Out vs_main(uint id : SV_VertexID) {
    float2 uv = float2((id << 1) & 2, id & 2);
    Out output; output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0); output.uv = uv; return output;
}
float4 ps_main(Out input) : SV_Target {
    uint layer = input.uv.x >= 0.5 ? 1 : 0;
    float2 uv = float2(frac(input.uv.x * 2.0), input.uv.y);
    float2 p = uv * 2.0 - 1.0;
    p /= 1.0 - 0.2 * length(p);
    uv = (p + 1.0) * 0.5;
    bool inside = all(uv >= 0.0) && all(uv <= 1.0);
    return inside ? views.Sample(view_sampler, float3(uv, layer)) : float4(0.0, 0.0, 0.0, 1.0);
}
