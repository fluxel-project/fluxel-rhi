#version 450
layout(push_constant) uniform Push { uint value; } pc;
layout(location=0) in flat uint in_value;
layout(location=0) out vec4 color;
void main(){ color=vec4(float(in_value)/255.0,float(pc.value)/255.0,0,1); }
