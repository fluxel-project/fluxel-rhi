//! Mixed-value combined depth/stencil stencil-copy output assertions.

use super::*;

fn clear_combined(
    device: &Device<WebGpu>,
    encoder: &mut CommandEncoder<WebGpu>,
    texture: &Texture<WebGpu>,
    desc: &TextureDesc,
    layer: u32,
    depth: f32,
    stencil: u32,
) {
    let view = device
        .create_view(
            texture,
            &TextureViewDesc {
                format: desc.format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH | TextureAspects::STENCIL,
                mips: 0..1,
                layers: layer..layer + 1,
            },
        )
        .unwrap();
    encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&view),
                depth: Some((LoadOp::Clear(depth), StoreOp::Store)),
                stencil: Some((LoadOp::Clear(stencil), StoreOp::Store)),
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap()
        .finish()
        .unwrap();
}

#[wasm_bindgen_test(async)]
async fn gpu_stencil_copy_bitplanes_cover_all_values_and_preserve_other_planes() {
    let d = open().await;
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [16, 16, 1],
        array_layers: 3,
        format: TextureFormat::Depth32FloatStencil8,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC
            | TextureUsage::COPY_DST
            | TextureUsage::RENDER_ATTACHMENT
            | TextureUsage::TEXTURE_BINDING,
        view_formats: vec![],
    };
    if !matches!(d.texture_support(&desc), Support::Supported(_)) {
        console_log!("SKIP depth32float-stencil8 mixed stencil copy");
        return;
    }
    let source = d.create_texture(&desc).unwrap();
    let target = d.create_texture(&desc).unwrap();
    let alias = d.create_texture(&desc).unwrap();
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    for layer in 0..desc.array_layers {
        clear_combined(&d, &mut e, &source, &desc, layer, 0.25, 0);
        clear_combined(&d, &mut e, &target, &desc, layer, 0.75, 91);
        clear_combined(&d, &mut e, &alias, &desc, layer, 0.5, 19);
    }
    // One distinct stencil byte per texel. The padded GPU row layout is also
    // intentional: the copy path sees every bit plane, not a CPU conversion.
    let mut pattern = vec![0u8; 16 * 256];
    for y in 0..16 {
        for x in 0..16 {
            pattern[y * 256 + x] = (y * 16 + x) as u8;
        }
    }
    e.write_texture(
        TextureRegion {
            texture: &source,
            mip: 0,
            origin: [0, 0, 1],
            extent: [16, 16, 1],
            aspect: TextureAspects::STENCIL,
        },
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 16,
        },
        &pattern,
    )
    .unwrap();
    e.write_texture(
        TextureRegion {
            texture: &alias,
            mip: 0,
            origin: [0, 0, 0],
            extent: [16, 16, 1],
            aspect: TextureAspects::STENCIL,
        },
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 16,
        },
        &pattern,
    )
    .unwrap();
    e.copy_texture(
        TextureRegion {
            texture: &source,
            mip: 0,
            origin: [0, 0, 1],
            extent: [16, 16, 1],
            aspect: TextureAspects::STENCIL,
        },
        TextureRegion {
            texture: &target,
            mip: 0,
            origin: [0, 0, 1],
            extent: [16, 16, 1],
            aspect: TextureAspects::STENCIL,
        },
    )
    .unwrap();
    // Same native texture, disjoint layers: stencil_copy must first snapshot
    // the source aspect so its render writes cannot feed later texture loads.
    e.copy_texture(
        TextureRegion {
            texture: &alias,
            mip: 0,
            origin: [0, 0, 0],
            extent: [16, 16, 1],
            aspect: TextureAspects::STENCIL,
        },
        TextureRegion {
            texture: &alias,
            mip: 0,
            origin: [0, 0, 1],
            extent: [16, 16, 1],
            aspect: TextureAspects::STENCIL,
        },
    )
    .unwrap();
    let mut tickets = Vec::new();
    for (layer, aspect) in [
        (1, TextureAspects::STENCIL),
        (0, TextureAspects::STENCIL),
        (1, TextureAspects::DEPTH),
    ] {
        let out = buffer(
            &d,
            16 * 256,
            BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            MemoryClass::Device,
        );
        e.texture_to_buffer(
            TextureRegion {
                texture: &target,
                mip: 0,
                origin: [0, 0, layer],
                extent: [16, 16, 1],
                aspect,
            },
            &out,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 16,
            },
        )
        .unwrap();
        tickets.push((e.readback_buffer(&out, 0..16 * 256).unwrap(), layer, aspect));
    }
    let alias_stencil = buffer(
        &d,
        16 * 256,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let alias_depth = buffer(
        &d,
        16 * 256,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    for (out, aspect) in [
        (&alias_stencil, TextureAspects::STENCIL),
        (&alias_depth, TextureAspects::DEPTH),
    ] {
        e.texture_to_buffer(
            TextureRegion {
                texture: &alias,
                mip: 0,
                origin: [0, 0, 1],
                extent: [16, 16, 1],
                aspect,
            },
            out,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 16,
            },
        )
        .unwrap();
    }
    let alias_stencil_ticket = e.readback_buffer(&alias_stencil, 0..16 * 256).unwrap();
    let alias_depth_ticket = e.readback_buffer(&alias_depth, 0..16 * 256).unwrap();
    let point = submit(&d, e).await;
    for (ticket, layer, aspect) in tickets {
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        for y in 0..16 {
            for x in 0..16 {
                match aspect {
                    TextureAspects::STENCIL => assert_eq!(
                        bytes[y * 256 + x],
                        if layer == 1 { (y * 16 + x) as u8 } else { 91 }
                    ),
                    TextureAspects::DEPTH => assert_eq!(
                        f32::from_le_bytes(
                            bytes[y * 256 + x * 4..y * 256 + x * 4 + 4]
                                .try_into()
                                .unwrap()
                        ),
                        0.75
                    ),
                    _ => unreachable!(),
                }
            }
        }
    }
    let alias_stencil_bytes = unsafe { d.read_after(alias_stencil_ticket, &point).await }
        .unwrap()
        .bytes;
    let alias_depth_bytes = unsafe { d.read_after(alias_depth_ticket, &point).await }
        .unwrap()
        .bytes;
    for y in 0..16 {
        for x in 0..16 {
            assert_eq!(alias_stencil_bytes[y * 256 + x], (y * 16 + x) as u8);
            assert_eq!(
                f32::from_le_bytes(
                    alias_depth_bytes[y * 256 + x * 4..y * 256 + x * 4 + 4]
                        .try_into()
                        .unwrap()
                ),
                0.5
            );
        }
    }
}
