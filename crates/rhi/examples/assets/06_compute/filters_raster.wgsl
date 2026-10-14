// WebGL2 visual fallback for the storage-texture compute example.
// The fragment entries implement the same kernels while preserving the
// browser backend's honest lack of compute and storage-texture capabilities.
@group(0) @binding(1) var image: texture_2d<f32>;
@group(0) @binding(2) var image_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

fn gray(position: vec2<i32>, extent: vec2<i32>) -> f32 {
    let clamped = clamp(position, vec2<i32>(0), extent - vec2<i32>(1));
    return dot(textureLoad(image, clamped, 0).rgb, vec3<f32>(0.33333334));
}

fn emboss(position: vec2<i32>, extent: vec2<i32>) -> f32 {
    let value = -gray(position + vec2<i32>(-1, -1), extent)
        - gray(position, extent)
        + 2.0 * gray(position + vec2<i32>(1, 1), extent);
    return clamp(value + 0.5, 0.0, 1.0);
}

fn edge_detect(position: vec2<i32>, extent: vec2<i32>) -> f32 {
    var sum = 0.0;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let center = x == 0 && y == 0;
            sum += gray(position + vec2<i32>(x, y), extent)
                * select(-0.125, 1.0, center);
        }
    }
    return clamp(sum / 0.1, 0.0, 1.0);
}

fn sharpen(position: vec2<i32>, extent: vec2<i32>) -> vec3<f32> {
    var color = vec3<f32>(0.0);
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let center = x == 0 && y == 0;
            let weight = select(-1.0, 9.0, center);
            let sample_position = clamp(
                position + vec2<i32>(x, y), vec2<i32>(0), extent - vec2<i32>(1));
            color += textureLoad(image, sample_position, 0).rgb * weight;
        }
    }
    return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn pixel_position(input: VertexOut) -> vec2<i32> {
    let extent = vec2<i32>(textureDimensions(image));
    return clamp(vec2<i32>(input.uv * vec2<f32>(extent)), vec2<i32>(0), extent - vec2<i32>(1));
}

@fragment fn emboss_fs(input: VertexOut) -> @location(0) vec4<f32> {
    let value = emboss(pixel_position(input), vec2<i32>(textureDimensions(image)));
    return vec4<f32>(value, value, value, 1.0);
}

@fragment fn edgedetect_fs(input: VertexOut) -> @location(0) vec4<f32> {
    let value = edge_detect(pixel_position(input), vec2<i32>(textureDimensions(image)));
    return vec4<f32>(value, value, value, 1.0);
}

@fragment fn sharpen_fs(input: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(sharpen(pixel_position(input), vec2<i32>(textureDimensions(image))), 1.0);
}
