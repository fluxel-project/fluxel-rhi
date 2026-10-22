//! Focused texture transfer and filtered-depth output checks.
use super::*;

#[wasm_bindgen_test(async)]
async fn gpu_depth_upload_buffer_stride_and_linear_blit_preserve_values() {
    let d = open().await;
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_DST | TextureUsage::COPY_SRC | TextureUsage::TEXTURE_BINDING,
        view_formats: vec![],
    };
    let a = d.create_texture(&desc).unwrap();
    let b = d.create_texture(&desc).unwrap();
    let small = d
        .create_texture(&TextureDesc {
            extent: [2, 2, 1],
            ..desc.clone()
        })
        .unwrap();
    let gradient: Vec<u8> = (0..16)
        .flat_map(|i| (i as f32 / 16.0).to_le_bytes())
        .collect();
    let mut strided = vec![0xee; 84];
    for row in 0..4 {
        strided[4 + row * 20..4 + row * 20 + 16]
            .copy_from_slice(&gradient[row * 16..row * 16 + 16]);
    }
    let upload = buffer(
        &d,
        84,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    e.write_texture(
        TextureRegion {
            texture: &a,
            mip: 0,
            origin: [0, 0, 0],
            extent: [4, 4, 1],
            aspect: TextureAspects::DEPTH,
        },
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 16,
            rows_per_image: 4,
        },
        &gradient,
    )
    .unwrap();
    e.write_buffer(&upload, 0, &strided).unwrap();
    e.buffer_to_texture(
        &upload,
        TexelCopyBufferLayout {
            offset: 4,
            bytes_per_row: 20,
            rows_per_image: 4,
        },
        TextureRegion {
            texture: &b,
            mip: 0,
            origin: [0, 0, 0],
            extent: [4, 4, 1],
            aspect: TextureAspects::DEPTH,
        },
    )
    .unwrap();
    e.blit(
        TextureRegion {
            texture: &a,
            mip: 0,
            origin: [0, 0, 0],
            extent: [4, 4, 1],
            aspect: TextureAspects::DEPTH,
        },
        TextureRegion {
            texture: &small,
            mip: 0,
            origin: [0, 0, 0],
            extent: [2, 2, 1],
            aspect: TextureAspects::DEPTH,
        },
        FilterMode::Linear,
    )
    .unwrap();
    let mut tickets = Vec::new();
    for (texture, width, height) in [(&a, 4, 4), (&b, 4, 4), (&small, 2, 2)] {
        let out = buffer(
            &d,
            u64::from(height) * 256,
            BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            MemoryClass::Device,
        );
        e.texture_to_buffer(
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [width, height, 1],
                aspect: TextureAspects::DEPTH,
            },
            &out,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: height,
            },
        )
        .unwrap();
        tickets.push((
            e.readback_buffer(&out, 0..u64::from(height) * 256).unwrap(),
            width,
            height,
        ));
    }
    let point = submit(&d, e).await;
    for (index, (ticket, width, height)) in tickets.into_iter().enumerate() {
        // SAFETY: one recording produced every ticket before the awaited point.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        for y in 0..height as usize {
            for x in 0..width as usize {
                let actual = f32::from_le_bytes(
                    bytes[y * 256 + x * 4..y * 256 + x * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                let expected = if index < 2 {
                    (y * 4 + x) as f32 / 16.0
                } else {
                    [2.5f32, 4.5, 10.5, 12.5][y * 2 + x] / 16.0
                };
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "texture {index} pixel {x},{y}: {actual} != {expected}"
                );
            }
        }
    }
}

#[wasm_bindgen_test(async)]
async fn gpu_color_transfer_repacking_keeps_odd_offset_rows() {
    let d = open().await;
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [3, 2, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let src = buffer(
        &d,
        64,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let dst = buffer(
        &d,
        64,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let input: Vec<u8> = (0..64).map(|i| i as u8).collect();
    let mut expected = vec![0xa5; 64];
    expected[3..15].copy_from_slice(&input[1..13]);
    expected[18..30].copy_from_slice(&input[14..26]);
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    e.write_buffer(&src, 0, &input).unwrap();
    e.write_buffer(&dst, 0, &[0xa5; 64]).unwrap();
    e.buffer_to_texture(
        &src,
        TexelCopyBufferLayout {
            offset: 1,
            bytes_per_row: 13,
            rows_per_image: 2,
        },
        TextureRegion {
            texture: &texture,
            mip: 0,
            origin: [0, 0, 0],
            extent: [3, 2, 1],
            aspect: TextureAspects::COLOR,
        },
    )
    .unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &texture,
            mip: 0,
            origin: [0, 0, 0],
            extent: [3, 2, 1],
            aspect: TextureAspects::COLOR,
        },
        &dst,
        TexelCopyBufferLayout {
            offset: 3,
            bytes_per_row: 15,
            rows_per_image: 2,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&dst, 0..64).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: every transfer and the readback are covered by the point.
    assert_eq!(
        unsafe { d.read_after(ticket, &point).await }.unwrap().bytes,
        expected
    );
}

#[wasm_bindgen_test(async)]
async fn gpu_combined_depth_stencil_single_aspect_copy_preserves_other_plane() {
    let d = open().await;
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
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
        console_log!("SKIP depth32float-stencil8");
        return;
    }
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    let mut tickets = Vec::new();
    for selected in [TextureAspects::DEPTH, TextureAspects::STENCIL] {
        let source = d.create_texture(&desc).unwrap();
        let target = d.create_texture(&desc).unwrap();
        for (texture, depth, stencil) in [(&source, 0.25, 37), (&target, 0.75, 91)] {
            let view = d
                .create_view(
                    texture,
                    &TextureViewDesc {
                        format: desc.format,
                        dimension: TextureViewDimension::D2,
                        aspects: TextureAspects::DEPTH | TextureAspects::STENCIL,
                        mips: 0..1,
                        layers: 0..1,
                    },
                )
                .unwrap();
            for aspects in [TextureAspects::DEPTH, TextureAspects::STENCIL] {
                d.create_view(
                    texture,
                    &TextureViewDesc {
                        format: desc.format,
                        dimension: TextureViewDimension::D2,
                        aspects,
                        mips: 0..1,
                        layers: 0..1,
                    },
                )
                .unwrap();
            }
            e.begin_raster(&RasterPassDesc {
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
                    width: 4,
                    height: 4,
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
        e.copy_texture(
            TextureRegion {
                texture: &source,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: selected,
            },
            TextureRegion {
                texture: &target,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: selected,
            },
        )
        .unwrap();
        for aspect in [TextureAspects::DEPTH, TextureAspects::STENCIL] {
            let out = buffer(
                &d,
                4 * 256,
                BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                MemoryClass::Device,
            );
            e.texture_to_buffer(
                TextureRegion {
                    texture: &target,
                    mip: 0,
                    origin: [0, 0, 0],
                    extent: [4, 4, 1],
                    aspect,
                },
                &out,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: 256,
                    rows_per_image: 4,
                },
            )
            .unwrap();
            tickets.push((
                e.readback_buffer(&out, 0..4 * 256).unwrap(),
                selected,
                aspect,
            ));
        }
    }
    let point = submit(&d, e).await;
    for (ticket, selected, aspect) in tickets {
        // SAFETY: the point covers both single-plane copies and all readbacks.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        for y in 0..4 {
            for x in 0..4 {
                if aspect == TextureAspects::DEPTH {
                    let value = f32::from_le_bytes(
                        bytes[y * 256 + x * 4..y * 256 + x * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    assert_eq!(
                        value,
                        if selected == TextureAspects::DEPTH {
                            0.25
                        } else {
                            0.75
                        }
                    );
                } else {
                    assert_eq!(
                        bytes[y * 256 + x],
                        if selected == TextureAspects::STENCIL {
                            37
                        } else {
                            91
                        }
                    );
                }
            }
        }
    }
}

#[wasm_bindgen_test(async)]
async fn gpu_clear_1d_and_nonrenderable_color_formats() {
    let d = open().await;
    let make = |dimension, format| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension,
            extent: [4, 1, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap()
    };
    let one = make(TextureDimension::D1, TextureFormat::Rgba8Unorm);
    let packed = make(TextureDimension::D2, TextureFormat::Rgb9e5Ufloat);
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    e.clear_texture(
        &one,
        0..1,
        0..1,
        TextureClearValue::Color(ClearColor::Float([1.0, 0.0, 0.0, 1.0])),
    )
    .unwrap();
    e.clear_texture(
        &packed,
        0..1,
        0..1,
        TextureClearValue::Color(ClearColor::Float([1.0, 0.0, 0.0, 1.0])),
    )
    .unwrap();
    let mut tickets = Vec::new();
    for texture in [&one, &packed] {
        let out = buffer(
            &d,
            256,
            BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            MemoryClass::Device,
        );
        e.texture_to_buffer(
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            &out,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 1,
            },
        )
        .unwrap();
        tickets.push(e.readback_buffer(&out, 0..256).unwrap());
    }
    let point = submit(&d, e).await;
    for (i, ticket) in tickets.into_iter().enumerate() {
        // SAFETY: point covers both copies and readbacks.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        for x in 0..4 {
            let v = u32::from_le_bytes(bytes[x * 4..x * 4 + 4].try_into().unwrap());
            assert_eq!(v, if i == 0 { 0xff0000ff } else { 0x80000100 });
        }
    }
}

#[wasm_bindgen_test(async)]
async fn gpu_combined_depth_array_copy_covers_both_layers() {
    let d = open().await;
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [2, 2, 1],
        array_layers: 2,
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
        console_log!("SKIP depth32float-stencil8");
        return;
    }
    let source = d.create_texture(&desc).unwrap();
    let target = d.create_texture(&desc).unwrap();
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    for texture in [&source, &target] {
        for layer in 0..2 {
            let view = d
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
            let value = if std::ptr::eq(texture, &source) {
                (layer + 1) as f32 / 4.0
            } else {
                0.75
            };
            e.begin_raster(&RasterPassDesc {
                label: None,
                colors: &[],
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(&view),
                    depth: Some((LoadOp::Clear(value), StoreOp::Store)),
                    stencil: Some((LoadOp::Clear(91), StoreOp::Store)),
                    depth_read_only: false,
                    stencil_read_only: false,
                }),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 2,
                    height: 2,
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
    }
    e.copy_texture(
        TextureRegion {
            texture: &source,
            mip: 0,
            origin: [0, 0, 0],
            extent: [2, 2, 2],
            aspect: TextureAspects::DEPTH,
        },
        TextureRegion {
            texture: &target,
            mip: 0,
            origin: [0, 0, 0],
            extent: [2, 2, 2],
            aspect: TextureAspects::DEPTH,
        },
    )
    .unwrap();
    let out = buffer(
        &d,
        1024,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    e.texture_to_buffer(
        TextureRegion {
            texture: &target,
            mip: 0,
            origin: [0, 0, 0],
            extent: [2, 2, 2],
            aspect: TextureAspects::DEPTH,
        },
        &out,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 2,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&out, 0..1024).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: the readback belongs to this completed submission.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for layer in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let offset = layer * 512 + y * 256 + x * 4;
                let got = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                assert_eq!(got, (layer + 1) as f32 / 4.0);
            }
        }
    }
}
