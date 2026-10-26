//! Native-MSAA standalone-resolve output assertions.
//!
//! The source is always a one-mip, one-layer native multisample texture: this
//! is the WebGPU shape accepted as a render-pass resolve attachment. The second
//! test uses both copy-only and attachment-only partial targets to cover the
//! backend's scratch-copy and shader paths without discarding its source.
use super::*;

fn texture(
    d: &Device<WebGpu>,
    extent: [u32; 3],
    samples: u32,
    usage: TextureUsage,
) -> Texture<WebGpu> {
    d.create_texture(&TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent,
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples,
        usage,
        view_formats: vec![],
    })
    .unwrap()
}

fn region<'a>(
    texture: &'a Texture<WebGpu>,
    origin: [u32; 3],
    extent: [u32; 3],
) -> TextureRegion<'a, WebGpu> {
    TextureRegion {
        texture,
        mip: 0,
        origin,
        extent,
        aspect: TextureAspects::COLOR,
    }
}

fn read_texture(
    d: &Device<WebGpu>,
    e: &mut CommandEncoder<WebGpu>,
    texture: &Texture<WebGpu>,
    extent: [u32; 3],
) -> ReadbackTicket<WebGpu> {
    let output = buffer(
        d,
        u64::from(extent[1]) * 256,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    e.texture_to_buffer(
        region(texture, [0, 0, 0], extent),
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: extent[1],
        },
    )
    .unwrap();
    e.readback_buffer(&output, 0..u64::from(extent[1]) * 256)
        .unwrap()
}

#[wasm_bindgen_test(async)]
async fn gpu_standalone_resolve_full_render_targets() {
    let d = open().await;
    let source = texture(
        &d,
        [4, 4, 1],
        4,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    let destination = texture(
        &d,
        [4, 4, 1],
        1,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    e.clear_texture(
        &source,
        0..1,
        0..1,
        TextureClearValue::Color(ClearColor::Float([1.0, 0.0, 0.0, 1.0])),
    )
    .unwrap();
    e.resolve(
        region(&source, [0, 0, 0], [4, 4, 1]),
        region(&destination, [0, 0, 0], [4, 4, 1]),
    )
    .unwrap();
    let ticket = read_texture(&d, &mut e, &destination, [4, 4, 1]);
    let point = submit(&d, e).await;
    // SAFETY: the resolve and readback copy precede the completed point.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for y in 0..4 {
        for x in 0..4 {
            assert_eq!(
                &bytes[y * 256 + x * 4..y * 256 + x * 4 + 4],
                &[255, 0, 0, 255],
                "pixel {x},{y}"
            );
        }
    }
}

#[wasm_bindgen_test(async)]
async fn gpu_standalone_resolve_subrect_preserves_copy_and_attachment_sentinels() {
    let d = open().await;
    let source = texture(
        &d,
        [4, 4, 1],
        4,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    let destination = texture(
        &d,
        [8, 8, 1],
        1,
        TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
    );
    let attachment_destination = texture(
        &d,
        [8, 8, 1],
        1,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    let sentinel = [0, 0, 255, 255];
    e.write_texture(
        region(&destination, [0, 0, 0], [8, 8, 1]),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 32,
            rows_per_image: 8,
        },
        &sentinel.repeat(64),
    )
    .unwrap();
    e.clear_texture(
        &source,
        0..1,
        0..1,
        TextureClearValue::Color(ClearColor::Float([0.0, 1.0, 0.0, 1.0])),
    )
    .unwrap();
    e.clear_texture(
        &attachment_destination,
        0..1,
        0..1,
        TextureClearValue::Color(ClearColor::Float([0.0, 0.0, 1.0, 1.0])),
    )
    .unwrap();
    e.resolve(
        region(&source, [1, 1, 0], [2, 2, 1]),
        region(&destination, [2, 3, 0], [2, 2, 1]),
    )
    .unwrap();
    // A second resolve in the same recording proves the first partial path did
    // not discard the source. This target has no COPY_DST usage, so it selects
    // the attachment-only shader path.
    e.resolve(
        region(&source, [1, 1, 0], [2, 2, 1]),
        region(&attachment_destination, [4, 1, 0], [2, 2, 1]),
    )
    .unwrap();
    let copy_ticket = read_texture(&d, &mut e, &destination, [8, 8, 1]);
    let attachment_ticket = read_texture(&d, &mut e, &attachment_destination, [8, 8, 1]);
    let point = submit(&d, e).await;
    // SAFETY: the upload, resolves, and readback copies precede this point.
    let copy_bytes = unsafe { d.read_after(copy_ticket, &point).await }
        .unwrap()
        .bytes;
    let attachment_bytes = unsafe { d.read_after(attachment_ticket, &point).await }
        .unwrap()
        .bytes;
    assert_subrect(&copy_bytes, 2..4, 3..5, sentinel, "copy-only target");
    assert_subrect(
        &attachment_bytes,
        4..6,
        1..3,
        sentinel,
        "attachment-only target",
    );
}

fn assert_subrect(
    bytes: &[u8],
    xs: std::ops::Range<usize>,
    ys: std::ops::Range<usize>,
    outside: [u8; 4],
    target: &str,
) {
    for y in 0..8 {
        for x in 0..8 {
            let expected = if xs.contains(&x) && ys.contains(&y) {
                [0, 255, 0, 255]
            } else {
                outside
            };
            assert_eq!(
                &bytes[y * 256 + x * 4..y * 256 + x * 4 + 4],
                &expected,
                "{target} pixel {x},{y}"
            );
        }
    }
}

#[wasm_bindgen_test(async)]
async fn gpu_standalone_resolve_rejects_native_nonresolvable_formats() {
    let d = open().await;
    for format in [
        TextureFormat::Rgba8Uint,
        TextureFormat::R32Float,
        TextureFormat::R16Unorm,
    ] {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 4,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        };
        if !matches!(d.texture_support(&desc), Support::Supported(_)) {
            console_log!("SKIP native MSAA admission {:?}", format);
            continue;
        }
        let source = d.create_texture(&desc).unwrap();
        let destination = d
            .create_texture(&TextureDesc { samples: 1, ..desc })
            .unwrap();
        let mut e = d.encoder(d.lanes()[0].id).unwrap();
        assert!(matches!(
            e.resolve(
                region(&source, [0, 0, 0], [4, 4, 1]),
                region(&destination, [0, 0, 0], [4, 4, 1])
            ),
            Err(Error::Unsupported(_))
        ));
    }
}
