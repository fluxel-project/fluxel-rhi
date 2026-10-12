struct Output { float4 position : SV_Position; };
Output vs_main(uint id : SV_VertexID) { float2 p = id == 0 ? float2(-1, -1) : (id == 1 ? float2(-1, 3) : float2(3, -1)); Output o; o.position = float4(p, 0.999, 1); return o; }
float hash(float2 p) { return frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453); }
float4 ps_main(float4 position : SV_Position) : SV_TARGET { float2 cell = floor(position.xy * .72); float star = step(.9982, hash(cell)); float glow = step(.99965, hash(cell + 19.7)); float horizon = saturate(position.y / 720.0); float3 sky = lerp(float3(.002,.003,.012), float3(.035,.075,.24), horizon * horizon); return float4(star.xxx + glow * float3(.25,.35,.7) + sky, 1); }
