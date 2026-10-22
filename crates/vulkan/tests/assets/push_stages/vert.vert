#version 450
layout(push_constant) uniform Push { uint value; } pc;
layout(location=0) out flat uint out_value;
void main(){ vec2 positions[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3)); gl_Position=vec4(positions[gl_VertexIndex],0,1); out_value=pc.value; }
