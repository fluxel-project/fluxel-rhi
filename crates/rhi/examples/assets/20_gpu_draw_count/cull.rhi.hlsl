// RHI ABI adaptation: generate pure draw arguments and a GPU count.
struct Scene { float4 velocity; float4 offset; float4 color; float4x4 projection; float4 padding[9]; };
StructuredBuffer<Scene> scenes : register(t0);
RWStructuredBuffer<uint4> draws : register(u1);
RWByteAddressBuffer count : register(u2);
[numthreads(1,1,1)]
void CSMain() {
    uint visible = 0;
    for (uint index = 0; index < 1024; ++index) {
        Scene scene = scenes[index];
        float4 left = mul(float4(-0.05, 0, 1, 1) + scene.offset, scene.projection); left /= left.w;
        float4 right = mul(float4(0.05, 0, 1, 1) + scene.offset, scene.projection); right /= right.w;
        if (-0.5 < right.x && left.x < 0.5) { draws[visible++] = uint4(3, 1, 0, index); }
    }
    count.Store(0, visible);
}
