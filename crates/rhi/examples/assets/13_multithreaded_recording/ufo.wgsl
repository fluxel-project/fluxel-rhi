struct Transform {
    mvp: mat4x4<f32>,
    tint: vec4<f32>,
};

@group(0) @binding(0) var<uniform> transform: Transform;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) _uv: vec2<f32>,
    @location(3) color: vec3<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) view: vec3<f32>,
    @location(3) light: vec3<f32>,
};

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    var output: VertexOut;
    let clip = transform.mvp * vec4<f32>(input.position, 1.0);
    output.position = clip;
    // Deliberately keep the C++ sample's clip-space normal transform.
    output.normal = mat3x3<f32>(
        transform.mvp[0].xyz,
        transform.mvp[1].xyz,
        transform.mvp[2].xyz,
    ) * input.normal;
    output.color = select(input.color, transform.tint.rgb, all(input.color == vec3<f32>(1.0, 0.0, 0.0)));
    output.light = -clip.xyz;
    output.view = -clip.xyz;
    return output;
}

@fragment
fn ps_main(input: VertexOut) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal);
    let light = normalize(input.light);
    let view = normalize(input.view);
    let reflected = reflect(-light, normal);
    let diffuse = max(dot(normal, light), 0.0) * input.color;
    let specular = pow(max(dot(reflected, view), 0.0), 8.0) * vec3<f32>(0.75);
    return vec4<f32>(diffuse + specular, 1.0);
}
