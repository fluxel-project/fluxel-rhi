use super::*;

const ROCK_WITHOUT_COLOR_0: &[u8] = include_bytes!("../../../examples/assets/models/rock01.gltf");

#[test]
fn missing_color_0_writes_opaque_white_vertex_colours() {
    let model = load_embedded_model(ROCK_WITHOUT_COLOR_0, LoadOptions::NONE)
        .expect("rock asset without COLOR_0 loads");

    assert!(!model.vertices.is_empty());
    for vertex in model.vertices.chunks_exact(VKGLTF_VERTEX_STRIDE) {
        let color = std::array::from_fn(|component| {
            let offset = 32 + component * 4;
            f32::from_le_bytes(vertex[offset..offset + 4].try_into().expect("colour bytes"))
        });
        assert_eq!(color, [1.0; 4]);
    }
}
