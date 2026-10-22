// Derived from Microsoft D3D12nBodyGravity ParticleDraw.hlsl (MIT).
// Expand the original geometry-shader strip in a vertex shader.
struct PosVelo { float4 pos; float4 velo; };
StructuredBuffer<PosVelo> particles : register(t0);
struct Varying { float4 pos : SV_Position; float2 tex : TEXCOORD0; float4 color : COLOR; };
Varying VSMain(uint vertex : SV_VertexID, uint instance : SV_InstanceID) {
    static const uint indices[6] = {0, 1, 2, 2, 1, 3};
    static const float2 corners[4] = {float2(-1,1),float2(1,1),float2(-1,-1),float2(1,-1)};
    static const float2 uv[4] = {float2(0,0),float2(1,0),float2(0,1),float2(1,1)};
    uint i = indices[vertex];
    float3 p = particles[instance].pos.xyz + float3(corners[i] * 10, 0);
    float z = p.z - 1500;
    float yscale = 1.0 / tan(0.4);
    Varying o;
    o.pos = float4(p.x * yscale / (1280.0/720.0), p.y * yscale, -z * (5000.0/4999.0) - 5000.0/4999.0, -z);
    o.tex = uv[i];
    o.color = lerp(float4(1,0.1,0.1,1), float4(1,1,0.2,1), particles[instance].velo.w / 9);
    return o;
}
float4 PSMain(Varying input) : SV_Target {
    float intensity = clamp(0.5 - length(float2(0.5,0.5) - input.tex), 0.0, 0.5) * 2.0;
    return float4(input.color.xyz, intensity);
}
