//! Android NativeActivity harness for real Vulkan and GLES RHI example captures.
//!
//! Each build selects one example workload and keeps its native window, device,
//! surface, and GPU resources on the Android event-loop thread.
#![allow(dead_code)]

#[cfg(feature = "demo14")]
#[path = "../../../../crates/rhi/examples/14_descriptor_indexing/main.rs"]
mod descriptor_indexing;

#[cfg(feature = "demo13")]
#[path = "../../../../crates/rhi/examples/13_multithreaded_recording/main.rs"]
mod multithreaded_recording;

#[cfg(feature = "demo12")]
#[path = "../../../../crates/rhi/examples/12_screenshot/main.rs"]
mod screenshot;

#[cfg(feature = "demo06")]
#[path = "../../../../crates/rhi/examples/06_compute/main.rs"]
mod compute;
#[cfg(feature = "demo04")]
#[path = "../../../../crates/rhi/examples/04_dynamic_uniform_buffer/main.rs"]
mod dynamic_uniform;
#[cfg(feature = "demo10")]
#[path = "../../../../crates/rhi/examples/10_indirect_draw/main.rs"]
mod indirect_draw;
#[cfg(feature = "demo03")]
#[path = "../../../../crates/rhi/examples/03_instancing/main.rs"]
mod instancing;
#[cfg(feature = "demo08")]
#[path = "../../../../crates/rhi/examples/08_msaa/main.rs"]
mod msaa;
#[cfg(feature = "demo11")]
#[path = "../../../../crates/rhi/examples/11_occlusion_query/main.rs"]
mod occlusion_query;
#[cfg(feature = "demo07")]
#[path = "../../../../crates/rhi/examples/07_offscreen/main.rs"]
mod offscreen;
#[cfg(feature = "demo05")]
#[path = "../../../../crates/rhi/examples/05_push_constants/main.rs"]
mod push_constants;
#[cfg(feature = "demo02")]
#[path = "../../../../crates/rhi/examples/02_texture/main.rs"]
mod texture;
#[cfg(feature = "demo09")]
#[path = "../../../../crates/rhi/examples/09_texture_mipmap/main.rs"]
mod texture_mipmap;
#[cfg(all(
    not(feature = "demo02"),
    not(feature = "demo03"),
    not(feature = "demo04"),
    not(feature = "demo05"),
    not(feature = "demo06"),
    not(feature = "demo07"),
    not(feature = "demo08"),
    not(feature = "demo09"),
    not(feature = "demo10"),
    not(feature = "demo11"),
    not(feature = "demo12"),
    not(feature = "demo13"),
    not(feature = "demo14")
))]
#[path = "../../../../crates/rhi/examples/01_triangle/main.rs"]
mod triangle;
#[path = "../../../../crates/rhi/examples/utils/mod.rs"]
mod utils;

use fluxel_rhi::api::*;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::Arc;

#[cfg(all(
    not(feature = "demo02"),
    not(feature = "demo03"),
    not(feature = "demo04"),
    not(feature = "demo05"),
    not(feature = "demo06"),
    not(feature = "demo07"),
    not(feature = "demo08"),
    not(feature = "demo09"),
    not(feature = "demo10"),
    not(feature = "demo11"),
    not(feature = "demo12"),
    not(feature = "demo13"),
    not(feature = "demo14")
))]
type Scene<B> = triangle::Triangle<B>;
#[cfg(feature = "demo02")]
type Scene<B> = texture::TextureExample<B>;
#[cfg(feature = "demo03")]
type Scene<B> = instancing::Instancing<B>;
#[cfg(feature = "demo04")]
type Scene<B> = dynamic_uniform::DynamicUniforms<B>;
#[cfg(feature = "demo05")]
type Scene<B> = push_constants::PushWorkload<B>;
#[cfg(feature = "demo06")]
type Scene<B> = compute::Compute<B>;
#[cfg(feature = "demo07")]
type Scene<B> = offscreen::Offscreen<B>;
#[cfg(feature = "demo08")]
type Scene<B> = msaa::Msaa<B>;
#[cfg(feature = "demo09")]
type Scene<B> = texture_mipmap::Tunnel<B>;
#[cfg(feature = "demo10")]
type Scene<B> = indirect_draw::Workload<B>;
#[cfg(feature = "demo11")]
type Scene<B> = occlusion_query::Workload<B>;
#[cfg(feature = "demo12")]
type Scene<B> = screenshot::Screenshot<B>;
#[cfg(all(feature = "demo13", feature = "vulkan"))]
type Scene<B> = multithreaded_recording::Multithreaded<B>;
#[cfg(all(feature = "demo13", feature = "gles"))]
type Scene<B> = multithreaded_recording::Sequential<B>;
#[cfg(feature = "demo14")]
type Scene<B> = descriptor_indexing::DescriptorIndexing<B>;

#[cfg(feature = "demo02")]
const TEXTURE_WGSL: &[u8] =
    include_bytes!("../../../../crates/rhi/examples/assets/02_texture/texture.wgsl");
#[cfg(feature = "demo02")]
const TEXTURE_KTX: &[u8] =
    include_bytes!("../../../../crates/rhi/examples/assets/textures/metalplate01_rgba.ktx");

#[cfg(feature = "demo02")]
fn texture_pixels() -> Result<(Vec<u8>, u32, u32)> {
    let bytes = TEXTURE_KTX;
    if bytes.len() < 68
        || &bytes[..12] != b"\xABKTX 11\xBB\r\n\x1A\n"
        || &bytes[12..16] != b"\x01\x02\x03\x04"
    {
        return Err(Error::Unsupported("KTX1 little-endian RGBA8 texture"));
    }
    let word = |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let (width, height) = (word(36), word(40));
    if word(24) != 0x1908 || word(28) != 0x8058 || width == 0 || height == 0 {
        return Err(Error::Unsupported("KTX1 RGBA8 texture"));
    }
    let offset = 64usize
        .checked_add(word(60) as usize)
        .ok_or(Error::InvalidRange)?;
    let size = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(Error::InvalidRange)?;
    if offset
        .checked_add(4 + size)
        .is_none_or(|end| end > bytes.len())
        || word(offset) as usize != size
    {
        return Err(Error::InvalidRange);
    }
    Ok((bytes[offset + 4..offset + 4 + size].to_vec(), width, height))
}
mod host;

// SAFETY: Android loads this exported entry by its fixed NativeActivity ABI.
#[unsafe(no_mangle)]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    host::run(app);
}
