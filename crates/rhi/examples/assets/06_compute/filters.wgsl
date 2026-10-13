// Three storage-texture filters matching compute_filters.hlsl.
@group(0) @binding(0) var source_image: texture_2d<f32>;
@group(0) @binding(1) var result_image: texture_storage_2d<rgba8unorm, write>;

fn gray(position: vec2<i32>) -> f32 {
    return dot(textureLoad(source_image, position, 0).rgb,
        vec3<f32>(0.33333334));
}

fn convolution(position: vec2<i32>, kernel: array<f32, 9>, divisor: f32, bias: f32) -> f32 {
    var sum = 0.0;
    var index = 0u;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            sum += gray(position + vec2<i32>(x, y)) * kernel[index];
            index += 1u;
        }
    }
    return clamp(sum / divisor + bias, 0.0, 1.0);
}

fn inside(position: vec2<u32>) -> bool {
    let extent = textureDimensions(result_image);
    return position.x < extent.x && position.y < extent.y;
}

@compute @workgroup_size(16, 16, 1)
fn emboss(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!inside(id.xy)) { return; }
    let kernel = array<f32, 9>(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0);
    let value = convolution(vec2<i32>(id.xy), kernel, 1.0, 0.5);
    textureStore(result_image, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}

@compute @workgroup_size(16, 16, 1)
fn edgedetect(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!inside(id.xy)) { return; }
    let kernel = array<f32, 9>(-0.125, -0.125, -0.125, -0.125, 1.0,
        -0.125, -0.125, -0.125, -0.125);
    let value = convolution(vec2<i32>(id.xy), kernel, 0.1, 0.0);
    textureStore(result_image, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}

@compute @workgroup_size(16, 16, 1)
fn sharpen(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!inside(id.xy)) { return; }
    let kernel = array<f32, 9>(-1.0, -1.0, -1.0, -1.0, 9.0,
        -1.0, -1.0, -1.0, -1.0);
    var color = vec3<f32>(0.0);
    var index = 0u;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            color += textureLoad(source_image, vec2<i32>(id.xy) + vec2<i32>(x, y), 0).rgb
                * kernel[index];
            index += 1u;
        }
    }
    textureStore(result_image, vec2<i32>(id.xy), vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
