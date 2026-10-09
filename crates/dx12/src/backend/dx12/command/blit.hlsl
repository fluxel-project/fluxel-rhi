// Backend-private full-screen triangle used by the explicit D3D12 TextureBlit
// route. Build-time packaging must compile this as vs_5_1/ps_5_1 (entries
// `vs_main` and `ps_main`) and embed the resulting bytecode next to the command
// lowering. The shader maps a destination pixel to the requested source rectangle
// through root constants; it deliberately samples LOD zero because the SRV names
// exactly one source mip.

Texture2D<float4> source_texture : register(t0);
SamplerState linear_sampler : register(s0);

cbuffer BlitConstants : register(b0) {
    float4 source_uv_rect; // xy=min, zw=max
};

struct VertexOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

VertexOut vs_main(uint vertex_id : SV_VertexID) {
    float2 p = float2((vertex_id == 2) ? 3.0 : -1.0,
                      (vertex_id == 1) ? 3.0 : -1.0);
    VertexOut result;
    result.position = float4(p, 0.0, 1.0);
    result.uv = float2((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return result;
}

float4 ps_main(VertexOut input) : SV_Target {
    return source_texture.SampleLevel(
        linear_sampler,
        lerp(source_uv_rect.xy, source_uv_rect.zw, input.uv),
        0.0);
}
