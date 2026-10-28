#version 450

layout(early_fragment_tests) in;

layout(set = 0, binding = 0, std430) buffer SideEffects {
    uint side_effects[];
};

layout(location = 0) out vec4 color;

void main() {
    uint pixel = uint(gl_FragCoord.y) * 8u + uint(gl_FragCoord.x);
    side_effects[pixel] = 1u;
    color = vec4(0.0, 1.0, 0.0, 1.0);
}
