// DXIL source for the plants, ground, and sky passes of example 10.
cbuffer Frame : register(b0) { column_major float4x4 projection; column_major float4x4 view; };
Texture2DArray albedo_array : register(t1); Texture2D albedo : register(t1); SamplerState sampler0 : register(s2);
struct PlantIn { float3 p:LOCATION0; float3 n:LOCATION1; float2 uv:LOCATION2; float3 c:LOCATION3; float3 ip:LOCATION4; float3 r:LOCATION5; float scale:LOCATION6; int layer:LOCATION7; };
struct MeshIn { float3 p:LOCATION0; float3 n:LOCATION1; float2 uv:LOCATION2; };
struct O { float4 p:SV_Position; float2 uv:TEXCOORD0; float3 n:NORMAL0; };
struct PlantOut { float4 p:SV_Position; float2 uv:TEXCOORD0; float3 n:NORMAL0; nointerpolation int layer:TEXCOORD1; };
PlantOut plants_vs(PlantIn i) { PlantOut o; float3 q=i.p*i.scale+i.ip; o.p=mul(projection,mul(view,float4(q,1))); o.uv=i.uv; o.n=i.n; o.layer=i.layer; return o; }
float4 plants_ps(PlantOut i):SV_Target { float4 x=albedo_array.Sample(sampler0,float3(i.uv,i.layer)); clip(x.a-.5); return x; }
O ground_vs(MeshIn i) { O o;o.p=mul(projection,mul(view,float4(i.p,1)));o.uv=i.uv*32;o.n=i.n;return o; }
float4 ground_ps(O i):SV_Target { return albedo.Sample(sampler0,i.uv); }
O sky_vs(MeshIn i) { O o;o.p=float4(i.p.xy,0,1);o.uv=i.uv;o.n=i.n;return o; }
float4 sky_ps(O i):SV_Target { return lerp(float4(.93,.9,.81,1),float4(.35,.5,1,1),saturate(i.uv.y)); }
