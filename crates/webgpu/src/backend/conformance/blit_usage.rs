//! Browser regression for sampled blits from a COPY_SRC-only source texture.

use super::*;

#[wasm_bindgen_test(async)]
async fn gpu_scaled_blit_accepts_copy_src_only_logical_source() {
    let d = open().await;
    let usage = TextureUsage::COPY_SRC | TextureUsage::COPY_DST;
    let src = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage,
            view_formats: vec![],
        })
        .unwrap();
    let dst = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage,
            view_formats: vec![],
        })
        .unwrap();
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    let color = [64, 128, 192, 255];
    e.write_texture(
        TextureRegion {
            texture: &src,
            mip: 0,
            origin: [0, 0, 0],
            extent: [2, 2, 1],
            aspect: TextureAspects::COLOR,
        },
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 8,
            rows_per_image: 2,
        },
        &color.repeat(4),
    )
    .unwrap();
    e.blit(
        TextureRegion {
            texture: &src,
            mip: 0,
            origin: [0, 0, 0],
            extent: [2, 2, 1],
            aspect: TextureAspects::COLOR,
        },
        TextureRegion {
            texture: &dst,
            mip: 0,
            origin: [0, 0, 0],
            extent: [4, 4, 1],
            aspect: TextureAspects::COLOR,
        },
        FilterMode::Linear,
    )
    .unwrap();
    let output = buffer(
        &d,
        4 * 256,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    e.texture_to_buffer(
        TextureRegion {
            texture: &dst,
            mip: 0,
            origin: [0, 0, 0],
            extent: [4, 4, 1],
            aspect: TextureAspects::COLOR,
        },
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 4,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&output, 0..1024).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: the completion point covers the source upload, blit and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for row in 0..4 {
        for column in 0..4 {
            let offset = row * 256 + column * 4;
            assert_eq!(&bytes[offset..offset + 4], &color);
        }
    }
}
