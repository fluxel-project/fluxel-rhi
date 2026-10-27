//! Browser GPU regression for padded indirect-count argument packing.

use super::*;

struct ChunkCapacityReset;
impl Drop for ChunkCapacityReset {
    fn drop(&mut self) {
        super::super::indirect_count::set_test_chunk_capacity(None);
    }
}

#[wasm_bindgen_test(async)]
async fn gpu_padded_indirect_count_packs_one_source_window_and_masks_count() {
    let _chunk_capacity_reset = ChunkCapacityReset;
    let d = open().await;
    assert!(d.capabilities().multi_draw_count);
    let shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(
                b"struct O { @builtin(position) p: vec4f } @vertex fn vs(@builtin(vertex_index) i: u32) -> O { let p = array<vec2f, 3>(vec2f(-1., -1.), vec2f(3., -1.), vec2f(-1., 3.)); var o: O; o.p = vec4f(p[i % 3u], 0., 1.); return o; } @fragment fn fs() -> @location(0) vec4f { return vec4f(1., 0., 0., 1.); }"
                    .as_slice(),
            ),
            entries: vec![
                ShaderEntryPoint {
                    name: "vs".into(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: "fs".into(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                },
            ],
            required_features: vec![],
        })
        .await
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: shader.clone(),
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: shader,
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![],
            color_targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: 15,
            })],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: u64::MAX,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let view = d
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    // Four prefix bytes make the source offset nonzero. Each 24-byte record
    // has 16 bytes of draw ABI followed by two padding words.
    let args = buffer(
        &d,
        64,
        BufferUsage::INDIRECT | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let count = buffer(
        &d,
        8,
        BufferUsage::INDIRECT | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let image = buffer(
        &d,
        1024,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let indexed_args = buffer(
        &d,
        48,
        BufferUsage::INDIRECT | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let indices = buffer(
        &d,
        12,
        BufferUsage::INDEX | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let mut words = vec![0xfeed_faceu32];
    words.extend([3, 1, 0, 0, 0xa5a5_a5a5, 0x5a5a_5a5a]);
    // This poison command would still cover the target if the GPU count mask
    // were bypassed after padded source gathering.
    words.extend([3, 1, 0, 0, 0xdead_beef, 0xcafe_babe]);
    let mut encoder = d.encoder(d.lanes()[0].id).unwrap();
    encoder
        .write_buffer(
            &args,
            0,
            &words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    encoder
        .write_buffer(
            &indexed_args,
            0,
            &[3u32, 1, 0, 0, 0, 0xa5a5_a5a5, 3, 1, 0, 0, 0, 0x5a5a_5a5a]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    encoder
        .write_buffer(
            &indices,
            0,
            &[0u32, 1, 2]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    encoder
        .write_buffer(
            &count,
            0,
            &[0xabab_ababu32, 1]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    let mut pass = encoder
        .begin_raster(&RasterPassDesc {
            label: Some("indirect count pass"),
            colors: &colors,
            depth_stencil: None,
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
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.push_debug_group("count expansion")
        .expect("raster-pass debug groups are supported");
    pass.insert_debug_marker("before indirect count")
        .expect("raster-pass debug markers are supported");
    // Two test-only chunks prove that the second dispatch uses `base=1`.
    super::super::indirect_count::set_test_chunk_capacity(Some(1));
    d.raster_extras()
        .unwrap()
        .multi_draw_count(&mut pass, &args, 4, &count, 4, 2, 24, false)
        .unwrap();
    pass.pop_debug_group().unwrap();
    pass.finish().unwrap();
    super::super::indirect_count::set_test_chunk_capacity(None);
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &image,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let visible_ticket = encoder.readback_buffer(&image, 0..1024).unwrap();
    // An indexed padded source uses the same gather kernel. GPU count zero
    // must suppress both forced chunks, leaving this second clear untouched.
    encoder
        .write_buffer(&count, 4, &0u32.to_le_bytes())
        .unwrap();
    let mut indexed = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
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
        .unwrap();
    indexed.set_pipeline(&pipeline).unwrap();
    indexed
        .set_index_buffer(&indices, 0..12, IndexFormat::Uint32)
        .unwrap();
    super::super::indirect_count::set_test_chunk_capacity(Some(1));
    d.raster_extras()
        .unwrap()
        .multi_draw_count(&mut indexed, &indexed_args, 0, &count, 4, 2, 24, true)
        .unwrap();
    indexed.finish().unwrap();
    super::super::indirect_count::set_test_chunk_capacity(None);
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &image,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&image, 0..1024).unwrap();
    // Count one must also execute a valid padded indexed command, proving the
    // gather preserved all five indexed argument words rather than only the
    // zero-count masking behavior above.
    encoder
        .write_buffer(&count, 4, &1u32.to_le_bytes())
        .unwrap();
    let mut indexed_visible = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
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
        .unwrap();
    indexed_visible.set_pipeline(&pipeline).unwrap();
    indexed_visible
        .set_index_buffer(&indices, 0..12, IndexFormat::Uint32)
        .unwrap();
    super::super::indirect_count::set_test_chunk_capacity(Some(1));
    d.raster_extras()
        .unwrap()
        .multi_draw_count(
            &mut indexed_visible,
            &indexed_args,
            0,
            &count,
            4,
            2,
            24,
            true,
        )
        .unwrap();
    indexed_visible.finish().unwrap();
    super::super::indirect_count::set_test_chunk_capacity(None);
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &image,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let indexed_visible_ticket = encoder.readback_buffer(&image, 0..1024).unwrap();
    let point = submit(&d, encoder).await;
    // SAFETY: the submission point covers the packed indirect draw and copy.
    let visible = unsafe { d.read_after(visible_ticket, &point).await }
        .unwrap()
        .bytes;
    let hidden = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let indexed_visible = unsafe { d.read_after(indexed_visible_ticket, &point).await }
        .unwrap()
        .bytes;
    assert_eq!(&visible[256 + 4..256 + 8], &[255, 0, 0, 255]);
    assert_eq!(&hidden[256 + 4..256 + 8], &[0, 0, 0, 255]);
    assert_eq!(&indexed_visible[256 + 4..256 + 8], &[255, 0, 0, 255]);
}
