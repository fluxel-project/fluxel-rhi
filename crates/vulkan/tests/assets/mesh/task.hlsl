struct Payload { uint value; };
groupshared Payload p;
[numthreads(1,1,1)]
void main() { p.value = 1; DispatchMesh(1,1,1,p); }
