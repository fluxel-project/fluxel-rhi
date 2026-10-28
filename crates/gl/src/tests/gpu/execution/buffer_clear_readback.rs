//! Buffer clear range validation and ordered aligned/unaligned ticket snapshots.
use super::*;

async fn buffer_fill(d: &Device<Gl>) {
    const SIZE: u64 = 16_408;
    const END: u64 = 16_400;
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: SIZE,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 212).unwrap();
    // SAFETY: buffer and device survive the completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    e.write_buffer(&buffer, 0, &vec![0xadu8; SIZE as usize])
        .unwrap();
    e.clear_buffer(&buffer, 8..END, 0x12345678).unwrap();
    let ticket = e.readback_buffer(&buffer, 0..SIZE).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: this point covers the fill and readback.
    let actual = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    assert_eq!(&actual[..8], &[0xad; 8]);
    for word in actual[8..END as usize].chunks_exact(4) {
        assert_eq!(word, &[0x78, 0x56, 0x34, 0x12]);
    }
    assert_eq!(&actual[END as usize..], &[0xad; 8]);
    for range in [1..8, 8..11, 8..SIZE + 4, 12..8] {
        let plan = common::single_commands_plan(&fixture, 213).unwrap();
        // SAFETY: the device and buffer remain live during this failed command.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin invalid commands"));
        assert!(matches!(
            run.commands().clear_buffer(&buffer, range, 0),
            Err(Error::InvalidRange)
        ));
    }
}

async fn readback_snapshot(d: &Device<Gl>) {
    for read_usage in [BufferUsage::COPY_SRC, BufferUsage::MAP_READ] {
        // Preserve both the aligned snapshot path and the position-sensitive,
        // unaligned byte-range regression formerly in the readback fixture.
        for unaligned in [false, true] {
            let (initial, range, overwrite_offset, replacement) = if unaligned {
                (
                    (0..64u8).map(|i| i.wrapping_mul(11)).collect::<Vec<_>>(),
                    7..39,
                    0,
                    vec![0xc7; 64],
                )
            } else {
                (vec![0x11; 32], 8..24, 8, vec![0x77; 16])
            };
            let buffer = d
                .create_buffer(&BufferDesc {
                    label: None,
                    size: initial.len() as u64,
                    usage: BufferUsage::COPY_DST | read_usage,
                    memory: MemoryClass::Device,
                })
                .unwrap();
            let fixture = common::ImmediateFixture {
                device: d,
                lane: d.lanes()[0].id,
            };
            let plan = common::single_commands_plan(&fixture, 214 + u64::from(unaligned)).unwrap();
            // SAFETY: the buffer and both tickets live through the checkpoint.
            let mut session =
                unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
            let mut run = session
                .begin_commands()
                .unwrap_or_else(|_| panic!("begin snapshot"));
            let e = run.commands();
            e.write_buffer(&buffer, 0, &initial).unwrap();
            let first = e.readback_buffer(&buffer, range.clone()).unwrap();
            e.write_buffer(&buffer, overwrite_offset, &replacement)
                .unwrap();
            let second = e.readback_buffer(&buffer, range.clone()).unwrap();
            assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
            let point = match session.finish() {
                ImmediateFinishOutcome::Issued(prefix) => {
                    prefix.checkpoints.into_iter().next().unwrap().point
                }
                _ => panic!("finish snapshot"),
            };
            d.wait_completion(&point).await.unwrap();
            // SAFETY: completion covers both snapshot tickets.
            let before = unsafe { d.read_after(first, &point).await }.unwrap().bytes;
            let after = unsafe { d.read_after(second, &point).await }.unwrap().bytes;
            assert_eq!(before, initial[range.start as usize..range.end as usize]);
            assert_eq!(
                after,
                vec![if unaligned { 0xc7 } else { 0x77 }; (range.end - range.start) as usize]
            );
        }
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_buffer_fill() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL PBO upload fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let d = block_on(open(native));
    block_on(buffer_fill(&d));
    block_on(readback_snapshot(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_buffer_fill() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(buffer_fill(&d));
    block_on(readback_snapshot(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_buffer_fill() {
    let d = webgl2_device([8, 8]).await;
    buffer_fill(&d).await;
    readback_snapshot(&d).await;
}
