struct V { float4 pos : SV_Position; };
struct Constants { float scale; };
[[vk::push_constant]] ConstantBuffer<Constants> params;
[[vk::binding(0,0)]] StructuredBuffer<float4> positions;
[outputtopology("triangle")]
[numthreads(1,1,1)]
void main(out vertices V verts[3], out indices uint3 triangles[1]) {
 SetMeshOutputCounts(3,1);
 for (uint i=0;i<3;i++) verts[i].pos=float4(positions[i].xy*params.scale,0,1);
 triangles[0]=uint3(0,1,2);
}
