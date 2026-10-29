//! Browser proof of the immediate WebGL2 screenshot and readback path.
#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]

#[path = "support/webgl2.rs"]
mod support;

#[path = "../examples/12_screenshot/main.rs"]
mod screenshot;

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn screenshot_renders_reads_back_and_presents_three_webgl2_frames() {
    let canvas = support::canvas(640, 480);
    let result = screenshot::run_webgl2_screenshot(canvas.clone(), 3).await;
    result.expect("12 must render, read back, and present three WebGL2 frames");
    support::assert_canvas_rendered(&canvas);
    support::assert_canvas_alpha_opaque(&canvas);
    canvas.remove();
}

#[wasm_bindgen_test(async)]
async fn texture_to_buffer_preserves_top_rows_offset_and_row_padding() {
    use fluxel_rhi::{
        api::*,
        gl::{Gl, GlInstance},
    };
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };

    static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);

    let canvas = support::canvas(16, 16);
    let native = GlInstance::from_webgl2_canvas(canvas.clone()).expect("WebGL2 instance");
    // SAFETY: this test keeps the canvas-backed native owner and instance in
    // scope until all immediate commands and completion observations finish.
    let instance = unsafe { Instance::<Gl>::from_native(InstanceId::allocate().unwrap(), native) };
    let adapter = instance.enumerate().unwrap().into_iter().next().unwrap();
    let device = instance
        .open(DeviceRequest {
            adapter: adapter.id,
            required_features: vec![],
        })
        .await
        .unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let texture = device
        .create_texture(&TextureDesc {
            label: Some("WebGL2 readback layout fixture".into()),
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST),
            view_formats: vec![],
        })
        .unwrap();
    let copied_texture = device
        .create_texture(&TextureDesc {
            label: Some("WebGL2 texture-copy destination fixture".into()),
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST),
            view_formats: vec![],
        })
        .unwrap();
    let buffer = device
        .create_buffer(&BufferDesc {
            label: Some("WebGL2 padded readback layout fixture".into()),
            size: 32,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let ordinal = NEXT_PLAN.fetch_add(1, Ordering::Relaxed);
    // SAFETY: the single command step is realizable on this device's GL lane.
    let plan = unsafe {
        SubmissionPlan::<Gl>::from_compiled(
            PlanId {
                device: device.id(),
                ordinal,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    // SAFETY: this browser test issues all commands immediately on the
    // canvas-owning thread and waits for this exact command-step completion.
    let mut session = unsafe { device.immediate(plan, CompletionRequest::End).unwrap() };
    let mut commands = match session.begin_commands() {
        Ok(commands) => commands,
        Err(_) => panic!("WebGL2 readback fixture command step"),
    };
    let texels = [
        255, 0, 0, 255, 0, 255, 0, 255, // upper row: red, green
        0, 0, 255, 255, 255, 255, 255, 255, // lower row: blue, white
    ];
    commands
        .commands()
        .write_texture(
            TextureRegion {
                texture: &texture,
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
            &texels,
        )
        .unwrap();
    // Leave a tiny draw scissor active to prove the private framebuffer copy
    // does not inherit state from a preceding raster operation.
    use wasm_bindgen::JsCast as _;
    let raw = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::WebGl2RenderingContext>()
        .unwrap();
    raw.enable(web_sys::WebGl2RenderingContext::SCISSOR_TEST);
    raw.scissor(0, 0, 1, 1);
    commands
        .commands()
        .write_buffer(&buffer, 0, &[0xAA; 32])
        .unwrap();
    commands
        .commands()
        .transition_resources(&[
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &texture,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_WRITE,
                    },
                    to: TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                },
                layout: Some((ImageLayout::Common, ImageLayout::CopySrc)),
                ownership: None,
            }),
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &copied_texture,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: TextureUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    to: TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_WRITE,
                    },
                },
                layout: Some((ImageLayout::Undefined, ImageLayout::CopyDst)),
                ownership: None,
            }),
        ])
        .unwrap();
    commands
        .commands()
        .copy_texture(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 1, 0],
                extent: [2, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            TextureRegion {
                texture: &copied_texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 1, 1],
                aspect: TextureAspects::COLOR,
            },
        )
        .unwrap();
    commands
        .commands()
        .copy_texture(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            TextureRegion {
                texture: &copied_texture,
                mip: 0,
                origin: [0, 1, 0],
                extent: [2, 1, 1],
                aspect: TextureAspects::COLOR,
            },
        )
        .unwrap();
    commands
        .commands()
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &copied_texture,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::CopyDst, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
    commands
        .commands()
        .texture_to_buffer(
            TextureRegion {
                texture: &copied_texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 2, 1],
                aspect: TextureAspects::COLOR,
            },
            &buffer,
            TexelCopyBufferLayout {
                offset: 4,
                bytes_per_row: 12,
                rows_per_image: 2,
            },
        )
        .unwrap();
    let ticket = commands.commands().readback_buffer(&buffer, 0..32).unwrap();
    assert!(matches!(commands.close(), ImmediateRunOutcome::Issued(_)));
    let prefix = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => prefix,
        _ => panic!("immediate readback fixture did not finish"),
    };
    let point = prefix.checkpoints.into_iter().next().unwrap().point;
    device.wait_completion(&point).await.unwrap();
    // SAFETY: `point` covers the only command step, including the texture copy
    // and the readback captured immediately afterwards.
    let data = unsafe { device.read_after(ticket, &point).await.unwrap() };
    let expected = [
        0xAA, 0xAA, 0xAA, 0xAA, // prefix before layout offset
        0, 0, 255, 255, 255, 255, 255, 255, // copied public lower row
        0, 0, 0, 0, // padding between rows
        255, 0, 0, 255, 0, 255, 0, 255, // copied public upper row
        0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, // untouched suffix
    ];
    assert_eq!(data.bytes, expected);
    canvas.remove();
}
