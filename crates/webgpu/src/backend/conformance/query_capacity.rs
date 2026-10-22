//! Occlusion slot exhaustion and pass continuation on a browser GPU.
use super::*;

#[wasm_bindgen_test(async)]
async fn gpu_repeated_occlusion_exceeds_native_pass_capacity() {
    let d = open().await;
    let queries = d.queries(QueryKind::Occlusion).unwrap();
    let set = queries
        .create(&QueryDesc {
            kind: QueryKind::Occlusion,
            count: 1,
        })
        .unwrap();
    let mut encoder = d.encoder(d.lanes()[0].id).unwrap();
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
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([1.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Store,
    })];
    queries.reset(&mut encoder, &set, 0..1).unwrap();
    let mut pass = encoder
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
            occlusion_query_set: Some(&set),
            secondary_inheritance: None,
        })
        .unwrap();
    for _ in 0..4097 {
        queries.begin_occlusion(&mut pass, &set, 0).unwrap();
        queries.end_occlusion(&mut pass).unwrap();
    }
    pass.finish().unwrap();
    let result = buffer(
        &d,
        256,
        BufferUsage::QUERY_RESOLVE | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    queries
        .resolve(&mut encoder, &set, 0..1, &result, 0)
        .unwrap();
    let ticket = encoder.readback_buffer(&result, 0..8).unwrap();
    let pixels = buffer(
        &d,
        1024,
        BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &pixels,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let pixel_ticket = encoder.readback_buffer(&pixels, 0..4).unwrap();
    let point = submit(&d, encoder).await;
    // SAFETY: the awaited submission covers both native pass segments and result reduction.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    assert_eq!(u64::from_le_bytes(bytes.try_into().unwrap()), 0);
    // SAFETY: the same completion covers the attachment readback.
    assert_eq!(
        unsafe { d.read_after(pixel_ticket, &point).await }
            .unwrap()
            .bytes,
        [255, 0, 0, 255]
    );
}
