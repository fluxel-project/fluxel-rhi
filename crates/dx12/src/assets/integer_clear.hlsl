cbuffer ClearValue : register(b0) { uint4 value; };
float4 vs(uint id : SV_VertexID) : SV_Position {
    float2 p = float2((id << 1) & 2, id & 2);
    return float4(p * float2(2, -2) + float2(-1, 1), 0, 1);
}
uint4 ps_uint() : SV_Target { return value; }
int4 ps_sint() : SV_Target { return asint(value); }
