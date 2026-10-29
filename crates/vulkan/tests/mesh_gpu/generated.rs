//! Multi-draw and count-generated mesh dispatch entry cases.

use super::scenario::{Dispatch, exercise};
use super::support::block_on;
use fluxel_rhi_core::api::{
    Device, Shader, ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage,
};
use fluxel_rhi_vulkan::Vulkan;
use std::sync::{Arc, OnceLock};

#[test]
fn mesh_multi_indirect_depth_readback() {
    exercise(Dispatch::Multi, false);
}

#[test]
fn mesh_indirect_count_depth_readback() {
    exercise(Dispatch::Count, false);
}

#[test]
fn mesh_indirect_count_below_max_depth_readback() {
    exercise(Dispatch::CountOne, false);
}

#[test]
fn mesh_indirect_count_zero_depth_readback() {
    exercise(Dispatch::CountZero, false);
}

#[test]
fn mesh_indirect_count_clamp_depth_readback() {
    exercise(Dispatch::CountClamped, false);
}

pub(super) fn fractional_pixel_shader(d: &Device<Vulkan>) -> Shader<Vulkan> {
    static PIXEL_SPIRV: OnceLock<Vec<u8>> = OnceLock::new();
    let bytes = PIXEL_SPIRV
        .get_or_init(|| {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "fluxel-mesh-indirect-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let source = dir.join("mesh_pixel.hlsl");
            let output = dir.join("mesh_pixel.spv");
            std::fs::write(
                &source,
                "float4 main() : SV_Target { return float4(0.25,0,0,1); }",
            )
            .unwrap();
            let result = std::process::Command::new("dxc")
                .args([
                    "-spirv",
                    "-fspv-target-env=vulkan1.2",
                    "-T",
                    "ps_6_0",
                    "-E",
                    "main",
                    "-Fo",
                ])
                .arg(&output)
                .arg(&source)
                .output()
                .expect("hardware test requires dxc in PATH");
            assert!(
                result.status.success(),
                "dxc: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            let bytes = std::fs::read(&output).unwrap();
            let _ = std::fs::remove_dir_all(&dir);
            bytes
        })
        .clone();
    block_on(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Fragment,
            workgroup_size: None,
        }],
        required_features: vec![],
    }))
    .unwrap()
}
