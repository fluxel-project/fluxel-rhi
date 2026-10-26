use super::shader_route;
use fluxel_rhi_core::api::{ShaderFormat, ShaderRoute};

#[test]
fn spirv_is_not_advertised_without_its_worker_asset() {
    assert!(matches!(
        shader_route(ShaderFormat::SpirV, false),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        shader_route(ShaderFormat::SpirV, true),
        ShaderRoute::Naga(_)
    ));
}
