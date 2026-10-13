struct Transform {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    model: mat4x4<f32>,
    light_position: vec4<f32>,
};
@group(0) @binding(0) var<uniform> transform: Transform;
@group(0) @binding(1) var reflection_image: texture_2d<f32>;
@group(0) @binding(2) var reflection_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) projected: vec4<f32>,
};

@vertex fn mirror_vs(@location(0) position: vec3<f32>) -> VertexOut {
    let projected = transform.projection * transform.view
        * transform.model * vec4<f32>(position, 1.0);
    var output: VertexOut;
    output.position = projected;
    output.projected = projected;
    return output;
}

@fragment fn mirror_ps(
    input: VertexOut,
    @builtin(front_facing) front_facing: bool,
) -> @location(0) vec4<f32> {
    var projected = input.projected.xy / input.projected.w * 0.5 + 0.5;
    projected.y = 1.0 - projected.y;
    var color = vec4<f32>(0.0);
    for (var x = -3; x <= 3; x += 1) {
        for (var y = -3; y <= 3; y += 1) {
            color += textureSample(reflection_image, reflection_sampler,
                projected + vec2<f32>(f32(x), f32(y)) / 512.0) / 49.0;
        }
    }
    return select(vec4<f32>(0.0, 0.0, 0.0, 1.0),
        vec4<f32>(color.rgb, 1.0), front_facing);
}
