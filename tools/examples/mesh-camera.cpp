// Reproduce the initial 1280x720 MeshletInstancing camera using DirectXMath.
#include <DirectXMath.h>
#include <cmath>
#include <cstdio>
#include <cstring>

int main() {
    using namespace DirectX;
    const float yaw = XM_PI;
    const XMMATRIX view = XMMatrixLookToRH(
        XMVectorSet(0, 5, 35, 0),
        XMVectorSet(sinf(yaw), 0, cosf(yaw), 0),
        XMVectorSet(0, 1, 0, 0));
    const XMMATRIX projection = XMMatrixPerspectiveFovRH(XM_PI / 3, 1280.f / 720.f, 1, 10000);
    const XMMATRIX matrices[] = {XMMatrixTranspose(view), XMMatrixTranspose(view * projection)};
    for (const auto& matrix : matrices) {
        XMFLOAT4X4 values;
        XMStoreFloat4x4(&values, matrix);
        unsigned bits[16];
        std::memcpy(bits, &values, sizeof(bits));
        for (unsigned value : bits) std::printf("0x%08x, ", value);
        std::puts("");
    }
}
