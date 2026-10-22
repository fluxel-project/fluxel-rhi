// Adapt the official per-draw CBV to per-instance vertex attributes.
struct Output { float4 position : SV_POSITION; float4 color : COLOR; };
Output VSMain(float4 position : LOCATION0, float4 offset : LOCATION1, float4 color : LOCATION2,
              float4 p0 : LOCATION3, float4 p1 : LOCATION4, float4 p2 : LOCATION5, float4 p3 : LOCATION6) {
    Output output; float4x4 projection = float4x4(p0, p1, p2, p3);
    output.position = mul(position + offset, transpose(projection));
    float intensity = saturate((4.0f - output.position.z) / 2.0f);
    output.color = float4(color.xyz * intensity, 1.0f); return output;
}
float4 PSMain(Output input) : SV_TARGET { return input.color; }
