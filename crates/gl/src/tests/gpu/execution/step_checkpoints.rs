//! Real immediate GL Steps checkpoints at exact command-close boundaries.

use super::*;

async fn steps(d: &Device<Gl>) {
    let lane = d.lanes()[0].id;
    let mut descriptor = BufferDesc {
        label: None,
        size: 4,
        usage: BufferUsage::COPY_DST,
        memory: MemoryClass::Device,
    };
    let max_size = match d.buffer_support(&descriptor) {
        Support::Supported(support) => support.max_size,
        Support::Unsupported(reason) => panic!("ordinary buffer unsupported: {reason}"),
    };
    descriptor.size = max_size.checked_add(1).unwrap();
    // Exceeding a reported addressing limit must reject before attempting an
    // impossible allocation; a subsequent real submission must remain usable.
    assert!(matches!(
        d.buffer_support(&descriptor),
        Support::Unsupported(_)
    ));
    assert!(d.create_buffer(&descriptor).is_err());
    let buffers = (0..3)
        .map(|_| {
            d.create_buffer(&BufferDesc {
                label: None,
                size: 4,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap()
        })
        .collect::<Vec<_>>();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 24024,
            },
            Arc::from(
                (0..3)
                    .map(|_| StepDesc {
                        lane,
                        kind: StepKind::Commands,
                        waits: vec![],
                    })
                    .collect::<Vec<_>>(),
            ),
        )
    };
    let requested = [plan.step(2), plan.step(0), plan.step(2), plan.step(1)];
    let mut session =
        unsafe { d.immediate(plan.clone(), CompletionRequest::Steps(Arc::from(requested))) }
            .unwrap();
    let mut tickets = Vec::new();
    for (index, buffer) in buffers.iter().enumerate() {
        let mut run = match session.begin_commands() {
            Ok(run) => run,
            Err(_) => panic!("Steps commands rejected"),
        };
        run.commands()
            .write_buffer(buffer, 0, &((index as u32 + 1) * 17).to_le_bytes())
            .unwrap();
        tickets.push(run.commands().readback_buffer(buffer, 0..4).unwrap());
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    }
    let prefix = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => prefix,
        _ => panic!("Steps finish"),
    };
    assert_eq!(prefix.checkpoints.len(), requested.len());
    for (checkpoint, step) in prefix.checkpoints.iter().zip(requested) {
        assert_eq!(checkpoint.covers.lane_frontiers, [(lane, step)]);
        d.wait_completion(&checkpoint.point).await.unwrap();
    }
    // SAFETY: each ticket belongs to its matching closed step and that step's
    // requested completion point was observed above.
    for index in 0..3 {
        let point = prefix
            .checkpoints
            .iter()
            .find(|checkpoint| checkpoint.covers.lane_frontiers[0].1 == plan.step(index))
            .unwrap();
        assert_eq!(
            unsafe { d.read_after(tickets.remove(0), &point.point).await }
                .unwrap()
                .bytes,
            ((index as u32 + 1) * 17).to_le_bytes()
        );
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_step_checkpoints() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL Steps checkpoints", 64, 64).unwrap()).unwrap();
    let d = block_on(open(
        unsafe { GlInstance::from_wgl_window(&window, [64, 64]) }.unwrap(),
    ));
    block_on(steps(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_step_checkpoints() {
    let d = block_on(open(GlInstance::gles_pbuffer([64, 64]).unwrap()));
    block_on(steps(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_step_checkpoints() {
    use wasm_bindgen::JsCast;
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(8);
    canvas.set_height(8);
    steps(&open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await).await;
}
