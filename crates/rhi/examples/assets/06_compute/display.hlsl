Texture2D image : register(t1); SamplerState image_sampler : register(s2);
struct In { float3 position:LOCATION0; float2 uv:LOCATION1; };
struct Out { float4 position:SV_Position; float2 uv:LOCATION0; };
Out vs_main(In input) { Out o; o.position=float4(input.position,1); o.uv=input.uv; return o; }
float4 ps_main(Out input):SV_Target { return image.Sample(image_sampler,input.uv); }
