cbuffer Transform:register(b0){float4x4 mvp;};
struct I{float3 p:LOCATION0;float3 n:LOCATION1;float2 uv:LOCATION2;float3 c:LOCATION3;};
struct P{float4 p:SV_Position;float3 n:NORMAL0;float3 c:COLOR0;float3 v:TEXCOORD0;};
P phong_vs(I i){P o;o.p=mul(mvp,float4(i.p,1));o.n=normalize(i.n);o.c=i.c;o.v=-i.p;return o;}
float4 phong_ps(P i):SV_Target{float3 n=normalize(i.n);float3 l=normalize(float3(-2,4,-3));float3 v=normalize(i.v);float3 r=reflect(-l,n);float d=max(dot(n,l),.15);float s=pow(max(dot(r,v),0),16)*.5;return float4(i.c*(.25+d)+s,1);}
Texture2D image:register(t1);SamplerState samp:register(s2);struct M{float4 p:SV_Position;float2 uv:TEXCOORD0;};M mirror_vs(I i){M o;o.p=mul(mvp,float4(i.p,1));o.uv=i.uv;return o;}float4 mirror_ps(M i):SV_Target{float4 c=0;[unroll]for(int x=-3;x<=3;x++)[unroll]for(int y=-3;y<=3;y++)c+=image.Sample(samp,i.uv+float2(x,y)/512.)/49.;return c;}
