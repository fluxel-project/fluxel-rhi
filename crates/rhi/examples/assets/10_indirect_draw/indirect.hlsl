// DXIL source for the plants, ground, and sky passes of example 10.
cbuffer Frame : register(b0) { column_major float4x4 projection; column_major float4x4 view; };
Texture2DArray albedo_array : register(t1); Texture2D albedo : register(t1); SamplerState sampler0 : register(s2);
struct PlantIn { float3 p:LOCATION0; float3 n:LOCATION1; float2 uv:LOCATION2; float3 c:LOCATION3; float3 ip:LOCATION4; float3 r:LOCATION5; float scale:LOCATION6; int layer:LOCATION7; };
struct MeshIn { float3 p:LOCATION0; float3 n:LOCATION1; float2 uv:LOCATION2; };
struct O { float4 p:SV_Position; float2 uv:TEXCOORD0; float3 n:NORMAL0; };
struct PlantOut { float4 p:SV_Position; float3 n:NORMAL0; float3 c:COLOR0; float2 uv:TEXCOORD0; float3 l:TEXCOORD1; nointerpolation int layer:TEXCOORD2; };
PlantOut plants_vs(PlantIn i) {
    PlantOut o;
    float4x4 mx, my, mz;
    float s = sin(i.r.x), c = cos(i.r.x);
    mx[0] = float4(c, s, 0, 0); mx[1] = float4(-s, c, 0, 0); mx[2] = float4(0, 0, 1, 0); mx[3] = float4(0, 0, 0, 1);
    s = sin(i.r.y); c = cos(i.r.y);
    my[0] = float4(c, 0, s, 0); my[1] = float4(0, 1, 0, 0); my[2] = float4(-s, 0, c, 0); my[3] = float4(0, 0, 0, 1);
    s = sin(i.r.z); c = cos(i.r.z);
    mz[0] = float4(1, 0, 0, 0); mz[1] = float4(0, c, s, 0); mz[2] = float4(0, -s, c, 0); mz[3] = float4(0, 0, 0, 1);
    float4x4 rotation = mul(mz, mul(my, mx));
    float4 position = mul(rotation, float4(i.p * i.scale + i.ip, 1));
    o.p = mul(projection, mul(view, position));
    o.n = mul((float3x3)rotation, i.n);
    o.c = i.c;
    o.uv = i.uv;
    o.l = float3(0, -5, 0) - position.xyz;
    o.layer = i.layer;
    return o;
}
float4 plants_ps(PlantOut i):SV_Target {
    float4 x = albedo_array.Sample(sampler0, float3(i.uv, i.layer));
    clip(x.a - .5);
    float3 diffuse = max(dot(normalize(i.n), normalize(i.l)), 0) * i.c;
    return float4((float3(.65, .65, .65) + diffuse) * x.rgb, 1);
}
O ground_vs(MeshIn i) { O o;o.p=mul(projection,mul(view,float4(i.p,1)));o.uv=i.uv*32;o.n=i.n;return o; }
float4 ground_ps(O i):SV_Target { return albedo.Sample(sampler0,i.uv); }
// The sky sphere stays centred on the camera: retain only the view rotation,
// then project it.  Treating model-space xy as clip coordinates flattened the
// sphere into the large ellipse visible in the DX12 port.
O sky_vs(MeshIn i) { O o; float4 p=mul(view,float4(i.p,0)); o.p=mul(projection,float4(p.xyz,1)); o.uv=i.uv; o.n=i.n; return o; }
float4 sky_ps(O i):SV_Target { return lerp(float4(.93,.9,.81,1),float4(.35,.5,1,1),saturate(i.uv.y)); }
