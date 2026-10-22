struct V { float4 p : SV_Position; nointerpolation float4 color : COLOR0; };
V vs(uint v : SV_VertexID, uint instance : SV_InstanceID) { V o; float x = (v << 1) & 2; float y = v & 2; o.p = float4(x * 2 - 1, 1 - y * 2, 0.5, 1); o.color = float4(v / 255.0, instance / 255.0, 0, 1); return o; }
float4 ps(V i) : SV_Target { return i.color; }
