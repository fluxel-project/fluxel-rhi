// Array-texture fragment stage split from instancing.hlsl for the rock path.
cbuffer Scene : register(b0) {
  column_major float4x4 projection;
  column_major float4x4 view;
  float4 light;
  float local_speed;
  float global_speed;
  float2 _padding;
};
Texture2DArray albedo : register(t1);
SamplerState linear_sampler : register(s2);

struct Out {
  float4 p: SV_Position;
  float3 n: NORMAL0;
  float3 c: COLOR0;
  float2 uv: TEXCOORD0;
  float3 v: TEXCOORD1;
  float3 l: TEXCOORD2;
  float layer: TEXCOORD3;
};

float4 rock_ps(Out i) : SV_Target {
  float3 base = albedo.Sample(linear_sampler, float3(i.uv, i.layer)).rgb * i.c;
  float3 n = normalize(i.n), l = normalize(i.l), v = normalize(i.v);
  float ndl = max(dot(n, l), 0);
  float3 diffuse = max(ndl, 0.1) * base;
  float3 spec = ndl > 0
    ? pow(max(dot(reflect(-l, n), v), 0), 16) * float3(.75, .75, .75) * base.r
    : 0;
  return float4(diffuse + spec, 1);
}
