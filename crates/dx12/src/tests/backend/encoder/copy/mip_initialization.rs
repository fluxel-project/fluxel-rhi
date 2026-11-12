//! Real-GPU regression for placed material mip initialization.
use super::*;
use std::{
    fs,
    mem::{MaybeUninit, size_of},
    path::PathBuf,
};

use windows::{
    Win32::Graphics::Direct3D12::{
        D3D12_MESSAGE, D3D12_MESSAGE_SEVERITY_CORRUPTION, D3D12_MESSAGE_SEVERITY_ERROR,
        D3D12_MESSAGE_SEVERITY_WARNING, ID3D12InfoQueue,
    },
    core::Interface,
};

fn debug_queue(device: &fluxel_rhi_core::api::Device<Dx12>) -> ID3D12InfoQueue {
    assert!(
        std::env::var_os("FLUXEL_DX12_DEBUG").is_some(),
        "this real-GPU regression requires FLUXEL_DX12_DEBUG=1"
    );
    // SAFETY: this backend-owned test borrows the matching device only while
    // it remains alive for the native diagnostics query.
    let native = unsafe { device.native_ref() };
    native
        .0
        .native
        .cast()
        .expect("debug-enabled device exposes ID3D12InfoQueue")
}

fn assert_no_native_errors(queue: &ID3D12InfoQueue) {
    // SAFETY: every queried index is below the current count. Aligned word
    // storage remains live through GetMessage and is large enough for bytes.
    let count = unsafe { queue.GetNumStoredMessagesAllowedByRetrievalFilter() };
    for index in 0..count {
        let mut bytes = 0;
        unsafe { queue.GetMessage(index, None, &mut bytes) }.unwrap();
        let words = bytes.div_ceil(size_of::<usize>());
        let mut storage = vec![MaybeUninit::<usize>::uninit(); words];
        unsafe {
            queue.GetMessage(
                index,
                Some(storage.as_mut_ptr().cast::<D3D12_MESSAGE>()),
                &mut bytes,
            )
        }
        .unwrap();
        let message = unsafe { &*storage.as_ptr().cast::<D3D12_MESSAGE>() };
        assert!(
            !matches!(
                message.Severity,
                D3D12_MESSAGE_SEVERITY_CORRUPTION
                    | D3D12_MESSAGE_SEVERITY_ERROR
                    | D3D12_MESSAGE_SEVERITY_WARNING
            ),
            "DX12 diagnostic {:?} ({:?})",
            message.Severity,
            message.ID
        );
    }
}

fn artifact_dir() -> Option<PathBuf> {
    std::env::var_os("FLUXEL_MIP_TEST_OUTPUT").map(PathBuf::from)
}

fn write_artifact(dir: &Option<PathBuf>, name: &str, bytes: &[u8]) {
    if let Some(dir) = dir {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join(name), bytes).unwrap();
    }
}

fn mip_texture_desc(label: &str, extent: [u32; 2], mip_levels: u32) -> TextureDesc {
    TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: fluxel_rhi_core::api::TextureFormat::Rgba8Unorm,
        mip_levels,
        samples: 1,
        usage: TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST),
        view_formats: vec![],
    }
}

fn color_region<'a>(
    texture: &'a fluxel_rhi_core::api::Texture<Dx12>,
    mip: u32,
    origin: [u32; 2],
    extent: [u32; 2],
) -> TextureRegion<'a, Dx12> {
    TextureRegion {
        texture,
        mip,
        origin: [origin[0], origin[1], 0],
        extent: [extent[0], extent[1], 1],
        aspect: TextureAspects::COLOR,
    }
}

fn transition_texture(
    encoder: &mut fluxel_rhi_core::api::CommandEncoder<Dx12>,
    texture: &fluxel_rhi_core::api::Texture<Dx12>,
    mips: std::ops::Range<u32>,
    from: ImageLayout,
    to: ImageLayout,
) {
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture,
                aspects: TextureAspects::COLOR,
                mips,
                layers: 0..1,
            },
            usage: StateTransition {
                from: match from {
                    ImageLayout::CopySrc => texture_uses(true),
                    ImageLayout::CopyDst => texture_uses(false),
                    ImageLayout::Undefined | ImageLayout::Common => texture_common(),
                    _ => unreachable!("mip regression uses only copy layouts"),
                },
                to: match to {
                    ImageLayout::CopySrc => texture_uses(true),
                    ImageLayout::CopyDst => texture_uses(false),
                    _ => unreachable!("mip regression writes only copy layouts"),
                },
            },
            layout: Some((from, to)),
            ownership: None,
        })])
        .unwrap();
}

fn write_rgba8_2d(bytes: &mut [u8], offset: usize, extent: [usize; 2], pixels: &[[u8; 4]]) {
    assert_eq!(pixels.len(), extent[0] * extent[1]);
    for y in 0..extent[1] {
        for x in 0..extent[0] {
            let byte = offset + y * 256 + x * 4;
            bytes[byte..byte + 4].copy_from_slice(&pixels[y * extent[0] + x]);
        }
    }
}

#[test]
fn generated_mips_initialize_placed_render_targets_and_preserve_partial_blits() {
    let (device, lane) = device_and_lane();
    let diagnostics = debug_queue(&device);
    // Device creation is outside this workload. Keep the diagnostic oracle
    // tied to the transitions and graphics blits below.
    unsafe { diagnostics.ClearStoredMessages() };
    let chain = device
        .create_texture(&mip_texture_desc("generated placed mip chain", [4, 4], 3))
        .unwrap();
    let source = device
        .create_texture(&mip_texture_desc("partial blit source", [2, 2], 1))
        .unwrap();
    let partial = device
        .create_texture(&mip_texture_desc("partial blit destination", [4, 4], 1))
        .unwrap();
    // Core usage has no attachment bit. The regression requires the native
    // RTV capability that DX12 allocates privately for graphics blit lowering.
    let native_chain = unsafe { chain.native_ref() };
    assert!(
        native_chain.0.backing_heap.is_some(),
        "regression requires placed allocation"
    );
    assert!(
        unsafe { native_chain.0.resource.GetDesc() }.Flags.contains(
            windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
        )
    );
    let upload = device
        .create_buffer(&BufferDesc {
            label: Some("mip initialization upload".into()),
            size: 2560,
            usage: BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("mip initialization readback".into()),
            size: 2048,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();

    let a = [16, 32, 64, 255];
    let b = [80, 48, 96, 255];
    let c = [32, 112, 128, 255];
    let d = [128, 160, 224, 255];
    let source_pixels = [
        [201, 3, 9, 255],
        [7, 211, 19, 255],
        [31, 41, 223, 255],
        [53, 67, 79, 255],
    ];
    let original = [[9, 19, 29, 255]; 16];
    let mut upload_bytes = [0u8; 2560];
    let mut chain_base = [[0u8; 4]; 16];
    for y in 0..4 {
        for x in 0..4 {
            chain_base[y * 4 + x] = match (x / 2, y / 2) {
                (0, 0) => a,
                (1, 0) => b,
                (0, 1) => c,
                _ => d,
            };
        }
    }
    write_rgba8_2d(&mut upload_bytes, 0, [4, 4], &chain_base);
    write_rgba8_2d(&mut upload_bytes, 1024, [2, 2], &source_pixels);
    write_rgba8_2d(&mut upload_bytes, 1536, [4, 4], &original);

    let mut expected_bytes = [0u8; 2048];
    write_rgba8_2d(&mut expected_bytes, 0, [2, 2], &[a, b, c, d]);
    write_rgba8_2d(&mut expected_bytes, 512, [1, 1], &[[64, 88, 128, 255]]);
    let mut expected_partial = original;
    for y in 0..2 {
        for x in 0..2 {
            expected_partial[(y + 1) * 4 + x + 1] = source_pixels[y * 2 + x];
        }
    }
    write_rgba8_2d(&mut expected_bytes, 1024, [4, 4], &expected_partial);
    let artifacts = artifact_dir();
    write_artifact(&artifacts, "upload.bin", &upload_bytes);
    write_artifact(&artifacts, "expected.bin", &expected_bytes);
    write_artifact(
        &artifacts,
        "readback-layout.txt",
        b"mip1 offset=0 bytes_per_row=256 rows=2; mip2 offset=512 bytes_per_row=256 rows=1; partial offset=1024 bytes_per_row=256 rows=4\n",
    );

    let mut encoder = device.encoder(lane).unwrap();
    encoder.write_buffer(&upload, 0, &upload_bytes).unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &upload,
                range: 0..2560,
            },
            usage: StateTransition {
                from: buffer_common(),
                to: copy_uses(true),
            },
            ownership: None,
        })])
        .unwrap();
    // This is the regression boundary: lower mips start Undefined and will be
    // internally used as render targets by the following linear blits.
    transition_texture(
        &mut encoder,
        &chain,
        0..3,
        ImageLayout::Undefined,
        ImageLayout::CopyDst,
    );
    transition_texture(
        &mut encoder,
        &source,
        0..1,
        ImageLayout::Undefined,
        ImageLayout::CopyDst,
    );
    transition_texture(
        &mut encoder,
        &partial,
        0..1,
        ImageLayout::Undefined,
        ImageLayout::CopyDst,
    );
    let rows_4 = TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: 256,
        rows_per_image: 4,
    };
    encoder
        .buffer_to_texture(&upload, rows_4, color_region(&chain, 0, [0, 0], [4, 4]))
        .unwrap();
    encoder
        .buffer_to_texture(
            &upload,
            TexelCopyBufferLayout {
                offset: 1024,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
            color_region(&source, 0, [0, 0], [2, 2]),
        )
        .unwrap();
    encoder
        .buffer_to_texture(
            &upload,
            TexelCopyBufferLayout {
                offset: 1536,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
            color_region(&partial, 0, [0, 0], [4, 4]),
        )
        .unwrap();
    transition_texture(
        &mut encoder,
        &chain,
        0..1,
        ImageLayout::CopyDst,
        ImageLayout::CopySrc,
    );
    encoder
        .blit(
            color_region(&chain, 0, [0, 0], [4, 4]),
            color_region(&chain, 1, [0, 0], [2, 2]),
            fluxel_rhi_core::api::FilterMode::Linear,
        )
        .unwrap();
    transition_texture(
        &mut encoder,
        &chain,
        1..2,
        ImageLayout::CopyDst,
        ImageLayout::CopySrc,
    );
    encoder
        .blit(
            color_region(&chain, 1, [0, 0], [2, 2]),
            color_region(&chain, 2, [0, 0], [1, 1]),
            fluxel_rhi_core::api::FilterMode::Linear,
        )
        .unwrap();
    transition_texture(
        &mut encoder,
        &chain,
        2..3,
        ImageLayout::CopyDst,
        ImageLayout::CopySrc,
    );
    transition_texture(
        &mut encoder,
        &source,
        0..1,
        ImageLayout::CopyDst,
        ImageLayout::CopySrc,
    );
    encoder
        .blit(
            color_region(&source, 0, [0, 0], [2, 2]),
            color_region(&partial, 0, [1, 1], [2, 2]),
            fluxel_rhi_core::api::FilterMode::Nearest,
        )
        .unwrap();
    transition_texture(
        &mut encoder,
        &partial,
        0..1,
        ImageLayout::CopyDst,
        ImageLayout::CopySrc,
    );
    encoder
        .texture_to_buffer(
            color_region(&chain, 1, [0, 0], [2, 2]),
            &readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
        )
        .unwrap();
    encoder
        .texture_to_buffer(
            color_region(&chain, 2, [0, 0], [1, 1]),
            &readback,
            TexelCopyBufferLayout {
                offset: 512,
                bytes_per_row: 256,
                rows_per_image: 1,
            },
        )
        .unwrap();
    encoder
        .texture_to_buffer(
            color_region(&partial, 0, [0, 0], [4, 4]),
            &readback,
            TexelCopyBufferLayout {
                offset: 1024,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..2048).unwrap();
    let point = submit(&device, lane, encoder.finish().unwrap());
    block_on(device.wait_completion(&point)).unwrap();
    let actual = unsafe { block_on(device.read_after(ticket, &point)).unwrap() }.bytes;
    write_artifact(&artifacts, "readback.bin", &actual);

    let expected_mip1 = [a, b, c, d];
    for (index, color) in expected_mip1.into_iter().enumerate() {
        let x = index % 2;
        let y = index / 2;
        assert_eq!(&actual[y * 256 + x * 4..y * 256 + x * 4 + 4], color);
    }
    assert_eq!(&actual[512..516], [64, 88, 128, 255]);
    for y in 0..4 {
        for x in 0..4 {
            let expected = expected_partial[y * 4 + x];
            let byte = 1024 + y * 256 + x * 4;
            assert_eq!(&actual[byte..byte + 4], expected);
        }
    }
    assert_no_native_errors(&diagnostics);
}
