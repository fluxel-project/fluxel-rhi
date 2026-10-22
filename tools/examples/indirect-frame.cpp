// Reproduce Microsoft's initial ExecuteIndirect animation and export frame 180.
#include <DirectXMath.h>
#include <cstdio>
#include <cstdlib>
using namespace DirectX;
struct Scene { XMFLOAT4 velocity, offset, color; XMFLOAT4X4 projection; float padding[36] = {}; };
float random_float(float min, float max) { float scale = static_cast<float>(rand()) / RAND_MAX; return scale * (max - min) + min; }
int main() {
    Scene scenes[1024] = {};
    for (auto& s : scenes) {
        s.velocity = XMFLOAT4(random_float(0.01f, 0.02f), 0, 0, 0);
        s.offset = XMFLOAT4(random_float(-5.0f, -1.5f), random_float(-1.0f, 1.0f), random_float(0.0f, 2.0f), 0);
        s.color = XMFLOAT4(random_float(0.5f, 1.0f), random_float(0.5f, 1.0f), random_float(0.5f, 1.0f), 1);
        XMStoreFloat4x4(&s.projection, XMMatrixTranspose(XMMatrixPerspectiveFovLH(XM_PIDIV4, 1280.f / 720.f, 0.01f, 20.0f)));
    }
    for (int frame = 0; frame < 180; ++frame) for (auto& s : scenes) {
        s.offset.x += s.velocity.x;
        if (s.offset.x > 2.5f) { s.velocity.x = random_float(0.01f, 0.02f); s.offset.x = -2.5f; }
    }
    FILE* file = nullptr; fopen_s(&file, "crates/rhi/examples/assets/20_gpu_draw_count/frame180.bin", "wb");
    if (!file) return 1;
    const bool ok = fwrite(scenes, sizeof(scenes), 1, file) == 1;
    fclose(file); return ok ? 0 : 1;
}
