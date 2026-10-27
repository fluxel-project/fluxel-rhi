#version 450
#extension GL_ARB_shader_draw_parameters : require

layout(location = 0) flat out int draw_id;

void main() {
    const vec2 positions[6] = vec2[6](
        vec2(-0.95, -0.9), vec2(-0.05, -0.9), vec2(-0.5, 0.9),
        vec2( 0.05, -0.9), vec2( 0.95, -0.9), vec2( 0.5, 0.9)
    );
    draw_id = gl_DrawIDARB;
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
