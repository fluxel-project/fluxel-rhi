#version 450

layout(location = 0) flat in int draw_id;
layout(location = 0) out vec4 color;

void main() {
    color = draw_id == 0
        ? vec4(1.0, 0.0, 0.0, 1.0)
        : vec4(0.0, 1.0, 0.0, 1.0);
}
