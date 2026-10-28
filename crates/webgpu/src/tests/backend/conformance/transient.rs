//! Native transient-usage probe plus resolve/readback on the browser GPU.
use super::*;

#[wasm_bindgen_test(async)]
async fn gpu_transient_attachment_clears_discards_and_resolves() {
    let d = open().await;
    let source = d
        .create_texture(&TextureDesc {
            label: Some("transient MSAA source".into()),
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 4,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TRANSIENT_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let native_usage = source.uses_native_transient_attachment();
    console_log!("WebGPU native TRANSIENT_ATTACHMENT usage applied: {native_usage}");
    let target = d
        .create_texture(&TextureDesc {
            label: Some("persistent resolve target".into()),
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let view = |texture: &Texture<WebGpu>| {
        d.create_view(
            texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap()
    };
    let source_view = view(&source);
    let target_view = view(&target);
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 1024,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut encoder = d.encoder(d.lanes()[0].id).unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&source_view),
        resolve_target: Some(AttachmentView::Texture(&target_view)),
        load: LoadOp::Clear(ClearColor::Float([1.0, 0.0, 0.0, 1.0])),
        store: StoreOp::Discard,
    })];
    encoder
        .begin_raster(&RasterPassDesc {
            label: Some("transient clear and resolve"),
            colors: &colors,
            depth_stencil: None,
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
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &target,
                mip: 0,
                origin: [0, 0, 0],
                extent: [16, 16, 1],
                aspect: TextureAspects::COLOR,
            },
            &output,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 64,
                rows_per_image: 16,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&output, 0..1024).unwrap();
    let point = submit(&d, encoder).await;
    // SAFETY: the resolve and readback copy precede this completion point.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for pixel in bytes.chunks_exact(4) {
        assert_eq!(pixel, [255, 0, 0, 255]);
    }
}
