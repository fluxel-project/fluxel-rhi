struct V { float4 pos : SV_Position; };
[outputtopology("triangle")]
[numthreads(1,1,1)]
void main(out vertices V verts[3], out indices uint3 triangles[1]) {
 SetMeshOutputCounts(3,1);
 verts[0].pos=float4(-1,-1,0,1); verts[1].pos=float4(0,1,0,1); verts[2].pos=float4(1,-1,0,1); triangles[0]=uint3(0,1,2);
}
