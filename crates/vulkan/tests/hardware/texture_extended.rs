//! Extended image transfer and clear GPU regressions.
use super::*;

#[test]
fn partial_upload_preserves_untouched_array_texels() {
    let device = device();
    let lane = device.lanes()[0].id;
    let image = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 3, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 48,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let full = || TextureRegion {
        texture: &image,
        mip: 0,
        origin: [0, 0, 1],
        extent: [4, 3, 1],
        aspect: TextureAspects::COLOR,
    };
    let mut e = device.encoder(lane).unwrap();
    e.write_texture(
        full(),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 16,
            rows_per_image: 3,
        },
        &[10; 48],
    )
    .unwrap();
    e.write_texture(
        TextureRegion {
            texture: &image,
            mip: 0,
            origin: [1, 1, 1],
            extent: [2, 1, 1],
            aspect: TextureAspects::COLOR,
        },
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 8,
            rows_per_image: 1,
        },
        &[99; 8],
    )
    .unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &image,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 1..2,
        },
        usage: StateTransition {
            from: copy_write(),
            to: copy_read(),
        },
        layout: Some((ImageLayout::Common, ImageLayout::CopySrc)),
        ownership: None,
    })])
    .unwrap();
    e.texture_to_buffer(
        full(),
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 16,
            rows_per_image: 3,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&output, 0..48).unwrap();
    let receipt = submit(&device, vec![e.finish().unwrap()], false);
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: the copy and readback have completed on this device.
    let data = block_on(unsafe { device.read_after(ticket, point) }).unwrap();
    let mut expected = vec![10; 48];
    expected[20..28].fill(99);
    assert_eq!(data.bytes, expected);
}

#[test]
fn compressed_blocks_round_trip_through_array_layers() {
    let device = device();
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: 2,
        format: TextureFormat::Bc1RgbaUnorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    if matches!(device.texture_support(&desc), Support::Unsupported(_)) {
        eprintln!("SKIP BC1 copy: device does not support requested texture");
        return;
    }
    let image = device.create_texture(&desc).unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let region = || TextureRegion {
        texture: &image,
        mip: 0,
        origin: [0, 0, 0],
        extent: [8, 8, 2],
        aspect: TextureAspects::COLOR,
    };
    let layout = || TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: 16,
        rows_per_image: 2,
    };
    let bytes: Vec<u8> = (0..64).collect();
    let mut e = device.encoder(device.lanes()[0].id).unwrap();
    e.write_texture(region(), layout(), &bytes).unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &image,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..2,
        },
        usage: StateTransition {
            from: copy_write(),
            to: copy_read(),
        },
        layout: Some((ImageLayout::Common, ImageLayout::CopySrc)),
        ownership: None,
    })])
    .unwrap();
    e.texture_to_buffer(region(), &output, layout()).unwrap();
    let ticket = e.readback_buffer(&output, 0..64).unwrap();
    let receipt = submit(&device, vec![e.finish().unwrap()], false);
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: the copy and readback completed on the ticket's device.
    let data = block_on(unsafe { device.read_after(ticket, point) }).unwrap();
    assert_eq!(data.bytes, bytes);
}

#[test]
fn integer_clear_covers_every_3d_slice() {
    let device = device();
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D3,
        extent: [4, 3, 2],
        array_layers: 1,
        format: TextureFormat::R32Uint,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    };
    if matches!(device.texture_support(&desc), Support::Unsupported(_)) {
        eprintln!("SKIP 3D integer clear: device does not support requested texture");
        return;
    }
    let image = device.create_texture(&desc).unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 96,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut e = device.encoder(device.lanes()[0].id).unwrap();
    let attachment = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let transition = |from, to, from_uses, to_uses| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &image,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: from_uses,
                to: to_uses,
            },
            layout: Some((from, to)),
            ownership: None,
        })
    };
    e.transition_resources(&[transition(
        ImageLayout::Undefined,
        ImageLayout::ColorAttachment,
        no_access(),
        attachment,
    )])
    .unwrap();
    e.clear_texture(
        &image,
        0..1,
        0..1,
        TextureClearValue::Color(ClearColor::Uint([0xdeadbeef, 0, 0, 0])),
    )
    .unwrap();
    e.transition_resources(&[transition(
        ImageLayout::ColorAttachment,
        ImageLayout::CopySrc,
        attachment,
        copy_read(),
    )])
    .unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &image,
            mip: 0,
            origin: [0, 0, 0],
            extent: [4, 3, 2],
            aspect: TextureAspects::COLOR,
        },
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 16,
            rows_per_image: 3,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&output, 0..96).unwrap();
    let receipt = submit(&device, vec![e.finish().unwrap()], false);
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: completion establishes host visibility of the readback.
    let data = block_on(unsafe { device.read_after(ticket, point) }).unwrap();
    assert_eq!(data.bytes, 0xdeadbeefu32.to_le_bytes().repeat(24));
}

#[test]
fn one_dimensional_integer_upload_readback() {
    let device = device();
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D1,
        extent: [5, 1, 1],
        array_layers: 1,
        format: TextureFormat::R16Uint,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    if matches!(device.texture_support(&desc), Support::Unsupported(_)) {
        eprintln!("SKIP 1D integer copy: device capability");
        return;
    }
    let image = device.create_texture(&desc).unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 10,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let region = || TextureRegion {
        texture: &image,
        mip: 0,
        origin: [0, 0, 0],
        extent: [5, 1, 1],
        aspect: TextureAspects::COLOR,
    };
    let layout = || TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: 10,
        rows_per_image: 1,
    };
    let bytes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut e = device.encoder(device.lanes()[0].id).unwrap();
    e.write_texture(region(), layout(), &bytes).unwrap();
    e.transition_resources(&[texture_transition(
        &image,
        ImageLayout::Common,
        ImageLayout::CopySrc,
        copy_write(),
        copy_read(),
    )])
    .unwrap();
    e.texture_to_buffer(region(), &output, layout()).unwrap();
    let ticket = e.readback_buffer(&output, 0..10).unwrap();
    let receipt = submit(&device, vec![e.finish().unwrap()], false);
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: completion establishes host visibility of the readback.
    let data = block_on(unsafe { device.read_after(ticket, point) }).unwrap();
    assert_eq!(data.bytes, bytes);
}

#[test]
fn blit_partial_rectangles_preserves_remaining_destination() {
    let device = device();
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    let source = device.create_texture(&desc).unwrap();
    let destination = device.create_texture(&desc).unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let layout = || TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: 16,
        rows_per_image: 4,
    };
    let input: Vec<u8> = (0..64).collect();
    let mut e = device.encoder(device.lanes()[0].id).unwrap();
    e.write_texture(region4(&source), layout(), &input).unwrap();
    e.write_texture(region4(&destination), layout(), &[0; 64])
        .unwrap();
    e.transition_resources(&[
        texture_transition(
            &source,
            ImageLayout::Common,
            ImageLayout::CopySrc,
            copy_write(),
            copy_read(),
        ),
        texture_transition(
            &destination,
            ImageLayout::Common,
            ImageLayout::CopyDst,
            copy_write(),
            copy_write(),
        ),
    ])
    .unwrap();
    e.blit(
        TextureRegion {
            texture: &source,
            mip: 0,
            origin: [1, 1, 0],
            extent: [2, 2, 1],
            aspect: TextureAspects::COLOR,
        },
        TextureRegion {
            texture: &destination,
            mip: 0,
            origin: [2, 0, 0],
            extent: [2, 2, 1],
            aspect: TextureAspects::COLOR,
        },
        FilterMode::Nearest,
    )
    .unwrap();
    e.transition_resources(&[texture_transition(
        &destination,
        ImageLayout::CopyDst,
        ImageLayout::CopySrc,
        copy_write(),
        copy_read(),
    )])
    .unwrap();
    e.texture_to_buffer(region4(&destination), &output, layout())
        .unwrap();
    let ticket = e.readback_buffer(&output, 0..64).unwrap();
    let receipt = submit(&device, vec![e.finish().unwrap()], false);
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: completion establishes host visibility of the readback.
    let data = block_on(unsafe { device.read_after(ticket, point) }).unwrap();
    let mut expected = vec![0; 64];
    expected[8..16].copy_from_slice(&input[20..28]);
    expected[24..32].copy_from_slice(&input[36..44]);
    assert_eq!(data.bytes, expected);
}

#[test]
fn multisample_resolve_readback() {
    let device = device();
    let make_desc = |samples, usage| TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples,
        usage,
        view_formats: vec![],
    };
    let ms_desc = make_desc(4, TextureUsage::RENDER_ATTACHMENT);
    let resolved_desc = make_desc(1, TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC);
    if matches!(device.texture_support(&ms_desc), Support::Unsupported(_)) {
        eprintln!("SKIP 4x resolve: device capability");
        return;
    }
    let source = device.create_texture(&ms_desc).unwrap();
    let destination = device.create_texture(&resolved_desc).unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();

    let color_write = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let color_read = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_READ,
    };
    let mut e = device.encoder(device.lanes()[0].id).unwrap();
    e.transition_resources(&[
        texture_transition(
            &source,
            ImageLayout::Undefined,
            ImageLayout::ColorAttachment,
            no_access(),
            color_write,
        ),
        texture_transition(
            &destination,
            ImageLayout::Undefined,
            ImageLayout::ResolveDst,
            no_access(),
            color_write,
        ),
    ])
    .unwrap();
    e.clear_texture(
        &source,
        0..1,
        0..1,
        TextureClearValue::Color(ClearColor::Float([1.0, 0.0, 0.0, 1.0])),
    )
    .unwrap();
    e.transition_resources(&[texture_transition(
        &source,
        ImageLayout::ColorAttachment,
        ImageLayout::ResolveSrc,
        color_write,
        color_read,
    )])
    .unwrap();
    e.resolve(region4(&source), region4(&destination)).unwrap();
    e.transition_resources(&[texture_transition(
        &destination,
        ImageLayout::ResolveDst,
        ImageLayout::CopySrc,
        color_write,
        copy_read(),
    )])
    .unwrap();
    e.texture_to_buffer(
        region4(&destination),
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 16,
            rows_per_image: 4,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&output, 0..64).unwrap();
    let receipt = submit(&device, vec![e.finish().unwrap()], false);
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: the completion point covers the readback.
    let data = block_on(unsafe { device.read_after(ticket, point) }).unwrap();
    assert_eq!(data.bytes, [255, 0, 0, 255].repeat(16));
}
