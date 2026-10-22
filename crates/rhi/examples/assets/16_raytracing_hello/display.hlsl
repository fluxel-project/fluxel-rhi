// Pixel-preserving presentation of the ray tracing output.
Texture2D<float4> output_texture : register(t0);

float4 vs_main(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}

float4 ps_main(float4 position : SV_Position) : SV_Target {
    return output_texture.Load(int3(uint2(position.xy), 0));
}
