cbuffer Scene : register(b0) {
  column_major float4x4 projection;
  column_major float4x4 view;
  float4 light;
  float local_speed;
  float global_speed;
  float2 _padding;
};
Texture2D albedo : register(t1);
SamplerState linear_sampler : register(s2);

struct MeshIn { float3 p: LOCATION0; float3 n: LOCATION1; float2 uv: LOCATION2; float3 c: LOCATION3; };
struct RockIn { float3 p: LOCATION0; float3 n: LOCATION1; float2 uv: LOCATION2; float3 c: LOCATION3; float3 ip: LOCATION4; float3 ir: LOCATION5; float scale: LOCATION6; float layer: LOCATION7; };
struct Out { float4 p: SV_Position; float3 n: NORMAL0; float3 c: COLOR0; float2 uv: TEXCOORD0; float3 v: TEXCOORD1; float3 l: TEXCOORD2; float layer: TEXCOORD3; };

float3x3 rotation(float3 a) {
  float x=a.x+local_speed, y=a.y+local_speed, z=a.z+local_speed;
  float3x3 mx=float3x3(1,0,0, 0,cos(x),sin(x), 0,-sin(x),cos(x));
  float3x3 my=float3x3(cos(y),0,-sin(y), 0,1,0, sin(y),0,cos(y));
  float3x3 mz=float3x3(cos(z),sin(z),0, -sin(z),cos(z),0, 0,0,1);
  return mul(mz,mul(my,mx));
}
float4x4 global_rotation() { float a=global_speed; return float4x4(cos(a),0,sin(a),0, 0,1,0,0, -sin(a),0,cos(a),0, 0,0,0,1); }
Out rock_vs(RockIn i) {
  Out o; float3x3 r=rotation(i.ir); float4x4 g=global_rotation();
  float3 world=mul(r,i.p)*i.scale+i.ip; float4 vp=mul(view,mul(g,float4(world,1)));
  o.p=mul(projection,vp); o.n=mul((float3x3)view,mul((float3x3)g,mul(r,i.n))); o.c=i.c; o.uv=i.uv;
  o.v=-vp.xyz; o.l=mul((float3x3)view,light.xyz)-vp.xyz; o.layer=i.layer; return o;
}
Out planet_vs(MeshIn i) {
  Out o; float4 vp=mul(view,float4(i.p,1)); o.p=mul(projection,vp); o.n=mul((float3x3)view,i.n); o.c=i.c; o.uv=i.uv; o.v=-vp.xyz; o.l=mul((float3x3)view,light.xyz)-vp.xyz; o.layer=0; return o;
}
float4 lit(Out i, float exponent, float intensity) : SV_Target {
  float3 base=albedo.Sample(linear_sampler,i.uv).rgb*i.c*intensity;
  float3 n=normalize(i.n), l=normalize(i.l), v=normalize(i.v);
  float ndl=max(dot(n,l),0); float3 diffuse=max(ndl,0.1)*base;
  float3 spec=ndl>0 ? pow(max(dot(reflect(-l,n),v),0),exponent)*float3(.75,.75,.75)*base.r : 0;
  return float4(diffuse+spec,1);
}
float4 rock_ps(Out i) : SV_Target { return lit(i,16,1); }
float4 planet_ps(Out i) : SV_Target { return lit(i,4,1.5); }

struct StarOut { float4 p: SV_Position; float3 uvw: TEXCOORD0; };
StarOut star_vs(uint id: SV_VertexID) { StarOut o; float2 uv=float2((id<<1)&2,id&2); o.uvw=float3(uv,uv.y); o.p=float4(uv*2-1,0,1); return o; }
float hash(float3 p) { p=frac(p*float3(443.897,441.423,437.195)); p+=dot(p,p.yxz+19.19); return frac((p.x+p.y)*p.z+(p.x+p.z)*p.y+(p.y+p.z)*p.x); }
float4 star_ps(StarOut i) : SV_Target { float r=hash(i.uvw); float s=r>=.99 ? pow((r-.99)/.01,16) : 0; return float4(s,s,s,1); }
