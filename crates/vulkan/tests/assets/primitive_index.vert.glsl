#version 450

void main() {
    const vec2 positions[6] = vec2[6](
        vec2(-0.95, -0.9), vec2(-0.05, -0.9), vec2(-0.5, 0.9),
        vec2( 0.05, -0.9), vec2( 0.95, -0.9), vec2( 0.5, 0.9)
    );
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
