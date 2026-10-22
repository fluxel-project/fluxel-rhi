//! Image plane and cross-format GPU regressions.
use super::*;

#[test]
fn maintenance5_copies_1d_array_to_3d_volume() {
    let device = device();
    let desc = |dimension, extent, layers| TextureDesc {
        label: None,
        dimension,
        extent,
        array_layers: layers,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    let source = device
        .create_texture(&desc(TextureDimension::D1, [4, 1, 1], 2))
        .unwrap();
    let target = device
        .create_texture(&desc(TextureDimension::D3, [4, 1, 2], 1))
        .unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let src = || TextureRegion {
        texture: &source,
        mip: 0,
        origin: [0, 0, 0],
        extent: [4, 1, 2],
        aspect: TextureAspects::COLOR,
    };
    let dst = || TextureRegion {
        texture: &target,
        mip: 0,
        origin: [0, 0, 0],
        extent: [4, 1, 2],
        aspect: TextureAspects::COLOR,
    };
    let layout = || TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: 16,
        rows_per_image: 1,
    };
    let bytes: Vec<u8> = (0..32).collect();
    let mut e = device.encoder(device.lanes()[0].id).unwrap();
    e.write_texture(src(), layout(), &bytes).unwrap();
    e.transition_resources(&[
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &source,
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
        }),
        texture_transition(
            &target,
            ImageLayout::Undefined,
            ImageLayout::CopyDst,
            no_access(),
            copy_write(),
        ),
    ])
    .unwrap();
    match e.copy_texture(src(), dst()) {
        Ok(()) => {}
        Err(Error::Unsupported(_)) => {
            eprintln!("SKIP maintenance5 copy: device capability");
            return;
        }
        Err(err) => panic!("copy failed: {err:?}"),
    }
    e.transition_resources(&[texture_transition(
        &target,
        ImageLayout::CopyDst,
        ImageLayout::CopySrc,
        copy_write(),
        copy_read(),
    )])
    .unwrap();
    e.texture_to_buffer(dst(), &output, layout()).unwrap();
    let ticket = e.readback_buffer(&output, 0..32).unwrap();
    let receipt = submit(&device, vec![e.finish().unwrap()], false);
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: completion establishes host visibility of the readback.
    let data = block_on(unsafe { device.read_after(ticket, point) }).unwrap();
    assert_eq!(data.bytes, bytes);
}

#[test]
fn copy_partial_compressed_mip_to_uncompressed_texel() {
    let device = device();
    let source_desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Bc1RgbaUnorm,
        mip_levels: 2,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    let target_desc = TextureDesc {
        extent: [1, 1, 1],
        mip_levels: 1,
        format: TextureFormat::Rg32Uint,
        ..source_desc.clone()
    };
    if matches!(
        device.texture_support(&source_desc),
        Support::Unsupported(_)
    ) || matches!(
        device.texture_support(&target_desc),
        Support::Unsupported(_)
    ) {
        eprintln!("SKIP partial compressed cross-format copy: device capability");
        return;
    }
    let source = device.create_texture(&source_desc).unwrap();
    let target = device.create_texture(&target_desc).unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 8,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let src = || TextureRegion {
        texture: &source,
        mip: 1,
        origin: [0, 0, 0],
        extent: [2, 2, 1],
        aspect: TextureAspects::COLOR,
    };
    let dst = || TextureRegion {
        texture: &target,
        mip: 0,
        origin: [0, 0, 0],
        extent: [1, 1, 1],
        aspect: TextureAspects::COLOR,
    };
    let layout = || TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: 8,
        rows_per_image: 1,
    };
    let input = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut e = device.encoder(device.lanes()[0].id).unwrap();
    e.write_texture(src(), layout(), &input).unwrap();
    e.transition_resources(&[
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &source,
                aspects: TextureAspects::COLOR,
                mips: 1..2,
                layers: 0..1,
            },
            usage: StateTransition {
                from: copy_write(),
                to: copy_read(),
            },
            layout: Some((ImageLayout::Common, ImageLayout::CopySrc)),
            ownership: None,
        }),
        texture_transition(
            &target,
            ImageLayout::Undefined,
            ImageLayout::CopyDst,
            no_access(),
            copy_write(),
        ),
    ])
    .unwrap();
    e.copy_texture(src(), dst()).unwrap();
    e.transition_resources(&[texture_transition(
        &target,
        ImageLayout::CopyDst,
        ImageLayout::CopySrc,
        copy_write(),
        copy_read(),
    )])
    .unwrap();
    e.texture_to_buffer(dst(), &output, layout()).unwrap();
    let ticket = e.readback_buffer(&output, 0..8).unwrap();
    let receipt = submit(&device, vec![e.finish().unwrap()], false);
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: the completed work owns the readback ticket.
    let data = block_on(unsafe { device.read_after(ticket, point) }).unwrap();
    assert_eq!(data.bytes, input);
}

#[test]
fn copy_compressed_blocks_to_uncompressed_texels() {
    let device = device();
    let desc = |extent, format| TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent,
        array_layers: 1,
        format,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    let source_desc = desc([8, 8, 1], TextureFormat::Bc1RgbaUnorm);
    let target_desc = desc([2, 2, 1], TextureFormat::Rg32Uint);
    if matches!(
        device.texture_support(&source_desc),
        Support::Unsupported(_)
    ) || matches!(
        device.texture_support(&target_desc),
        Support::Unsupported(_)
    ) {
        eprintln!("SKIP compressed cross-format copy: device capability");
        return;
    }
    let source = device.create_texture(&source_desc).unwrap();
    let target = device.create_texture(&target_desc).unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let src = || TextureRegion {
        texture: &source,
        mip: 0,
        origin: [0, 0, 0],
        extent: [8, 8, 1],
        aspect: TextureAspects::COLOR,
    };
    let dst = || TextureRegion {
        texture: &target,
        mip: 0,
        origin: [0, 0, 0],
        extent: [2, 2, 1],
        aspect: TextureAspects::COLOR,
    };
    let layout = || TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: 16,
        rows_per_image: 2,
    };
    let input: Vec<u8> = (0..32).collect();
    let mut e = device.encoder(device.lanes()[0].id).unwrap();
    e.write_texture(src(), layout(), &input).unwrap();
    e.transition_resources(&[
        texture_transition(
            &source,
            ImageLayout::Common,
            ImageLayout::CopySrc,
            copy_write(),
            copy_read(),
        ),
        texture_transition(
            &target,
            ImageLayout::Undefined,
            ImageLayout::CopyDst,
            no_access(),
            copy_write(),
        ),
    ])
    .unwrap();
    e.copy_texture(src(), dst()).unwrap();
    e.transition_resources(&[texture_transition(
        &target,
        ImageLayout::CopyDst,
        ImageLayout::CopySrc,
        copy_write(),
        copy_read(),
    )])
    .unwrap();
    e.texture_to_buffer(dst(), &output, layout()).unwrap();
    let ticket = e.readback_buffer(&output, 0..32).unwrap();
    let receipt = submit(&device, vec![e.finish().unwrap()], false);
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: completion establishes host visibility of the readback.
    let data = block_on(unsafe { device.read_after(ticket, point) }).unwrap();
    assert_eq!(data.bytes, input);
}

#[test]
fn copy_compatible_formats_from_array_to_volume() {
    let device = device();
    let desc = |dimension, layers, format| TextureDesc {
        label: None,
        dimension,
        extent: if dimension == TextureDimension::D3 {
            [2, 2, 2]
        } else {
            [2, 2, 1]
        },
        array_layers: layers,
        format,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    let source = device
        .create_texture(&desc(TextureDimension::D2, 2, TextureFormat::Rgba8Unorm))
        .unwrap();
    let target = device
        .create_texture(&desc(TextureDimension::D3, 1, TextureFormat::R32Uint))
        .unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let src = || TextureRegion {
        texture: &source,
        mip: 0,
        origin: [0, 0, 0],
        extent: [2, 2, 2],
        aspect: TextureAspects::COLOR,
    };
    let dst = || TextureRegion {
        texture: &target,
        mip: 0,
        origin: [0, 0, 0],
        extent: [2, 2, 2],
        aspect: TextureAspects::COLOR,
    };
    let layout = || TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: 8,
        rows_per_image: 2,
    };
    let input: Vec<u8> = (0..32).collect();
    let mut e = device.encoder(device.lanes()[0].id).unwrap();
    e.write_texture(src(), layout(), &input).unwrap();
    let barrier = |texture, layers, from, to, before, after| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers,
            },
            usage: StateTransition {
                from: before,
                to: after,
            },
            layout: Some((from, to)),
            ownership: None,
        })
    };
    e.transition_resources(&[
        barrier(
            &source,
            0..2,
            ImageLayout::Common,
            ImageLayout::CopySrc,
            copy_write(),
            copy_read(),
        ),
        barrier(
            &target,
            0..1,
            ImageLayout::Undefined,
            ImageLayout::CopyDst,
            no_access(),
            copy_write(),
        ),
    ])
    .unwrap();
    e.copy_texture(src(), dst()).unwrap();
    e.transition_resources(&[barrier(
        &target,
        0..1,
        ImageLayout::CopyDst,
        ImageLayout::CopySrc,
        copy_write(),
        copy_read(),
    )])
    .unwrap();
    e.texture_to_buffer(dst(), &output, layout()).unwrap();
    let ticket = e.readback_buffer(&output, 0..32).unwrap();
    let receipt = submit(&device, vec![e.finish().unwrap()], false);
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: completion establishes host visibility of the readback.
    let data = block_on(unsafe { device.read_after(ticket, point) }).unwrap();
    assert_eq!(data.bytes, input);
}

#[test]
fn depth_and_stencil_planes_round_trip() {
    let device = device();
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [2, 2, 1],
        array_layers: 1,
        format: TextureFormat::Depth32FloatStencil8,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    if matches!(device.texture_support(&desc), Support::Unsupported(_)) {
        eprintln!("SKIP depth/stencil copy: device capability");
        return;
    }
    let image = device.create_texture(&desc).unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 20,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let region = |aspect| TextureRegion {
        texture: &image,
        mip: 0,
        origin: [0, 0, 0],
        extent: [2, 2, 1],
        aspect,
    };
    let aspects = TextureAspects::DEPTH | TextureAspects::STENCIL;
    let barrier = |from, to, from_uses, to_uses| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &image,
                aspects,
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
    let depth = 0.5f32.to_le_bytes().repeat(4);
    let stencil = [1, 3, 5, 7];
    let mut e = device.encoder(device.lanes()[0].id).unwrap();
    e.transition_resources(&[barrier(
        ImageLayout::Undefined,
        ImageLayout::Common,
        no_access(),
        copy_write(),
    )])
    .unwrap();
    e.write_texture(
        region(TextureAspects::DEPTH),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 8,
            rows_per_image: 2,
        },
        &depth,
    )
    .unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &image,
            aspects: TextureAspects::DEPTH,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition {
            from: copy_write(),
            to: copy_read(),
        },
        layout: None,
        ownership: None,
    })])
    .unwrap();
    e.write_texture(
        region(TextureAspects::STENCIL),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 2,
            rows_per_image: 2,
        },
        &stencil,
    )
    .unwrap();
    e.transition_resources(&[barrier(
        ImageLayout::Common,
        ImageLayout::CopySrc,
        copy_write(),
        copy_read(),
    )])
    .unwrap();
    e.texture_to_buffer(
        region(TextureAspects::DEPTH),
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 8,
            rows_per_image: 2,
        },
    )
    .unwrap();
    e.texture_to_buffer(
        region(TextureAspects::STENCIL),
        &output,
        TexelCopyBufferLayout {
            offset: 16,
            bytes_per_row: 2,
            rows_per_image: 2,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&output, 0..20).unwrap();
    let receipt = submit(&device, vec![e.finish().unwrap()], false);
    let point = &receipt.checkpoints[0].point;
    block_on(device.wait_completion(point)).unwrap();
    // SAFETY: completion establishes host visibility of both copies.
    let data = block_on(unsafe { device.read_after(ticket, point) }).unwrap();
    assert_eq!(&data.bytes[..16], depth.as_slice());
    assert_eq!(&data.bytes[16..], &stencil);
}
