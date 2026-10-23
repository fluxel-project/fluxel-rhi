//! Core explicit-transition range validation and atomic batch rejection.
use super::*;

async fn transition_validation(d: &Device<Gl>) {
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let uses = |shader_write: bool| StateTransition {
        from: if shader_write {
            BufferUses {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::SHADER_WRITE,
            }
        } else {
            BufferUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            }
        },
        to: BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        },
    };
    let buffer_barrier = |range, shader_write| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range,
            },
            usage: uses(shader_write),
            ownership: None,
        })
    };
    let texture_barrier = |mips, layers, aspects| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &texture,
                aspects,
                mips,
                layers,
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
            layout: None,
            ownership: None,
        })
    };
    let valid_shader = cfg!(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl"
    ));
    // Each invalid operation runs on a fresh encoder: the immediate path poisons it on error.
    for index in 0..11 {
        let plan = common::single_commands_plan(&fixture, 200 + index as u64).unwrap();
        // SAFETY: all target resources remain borrowed and live through the session.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin transition fixture"));
        let invalid = match index {
            0 => buffer_barrier(8..4, false),
            1 => buffer_barrier(4..4, false),
            2 => buffer_barrier(0..65, false),
            3 => texture_barrier(1..0, 0..1, TextureAspects::COLOR),
            4 => texture_barrier(1..1, 0..1, TextureAspects::COLOR),
            5 => texture_barrier(0..3, 0..1, TextureAspects::COLOR),
            6 => texture_barrier(0..1, 1..0, TextureAspects::COLOR),
            7 => texture_barrier(0..1, 1..1, TextureAspects::COLOR),
            8 => texture_barrier(0..1, 0..3, TextureAspects::COLOR),
            9 => texture_barrier(0..1, 0..1, TextureAspects::DEPTH),
            _ => texture_barrier(0..1, 0..1, TextureAspects(0)),
        };
        let result = run
            .commands()
            .transition_resources(&[buffer_barrier(0..64, valid_shader), invalid]);
        assert!(
            matches!(result, Err(Error::InvalidRange)),
            "invalid case {index}: {result:?}"
        );
        match run.close() {
            ImmediateRunOutcome::Failed { error: _, prefix } => assert!(
                prefix.steps.is_empty() && prefix.checkpoints.is_empty(),
                "invalid batch issued a prefix"
            ),
            _ => panic!("invalid case {index}: expected clean failure"),
        }
    }
    let plan = common::single_commands_plan(&fixture, 220).unwrap();
    // SAFETY: resources and context outlive the valid session.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin valid fixture"));
    run.commands()
        .transition_resources(&[
            buffer_barrier(0..64, valid_shader),
            texture_barrier(0..2, 0..2, TextureAspects::COLOR),
        ])
        .unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_transition_validation() {
    use fluxel_host::{Window, WindowConfig};
    use std::{
        future::Future,
        task::{Context, Poll, Wake, Waker},
    };
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(transition_validation(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_transition_validation() {
    use std::{
        future::Future,
        task::{Context, Poll, Wake, Waker},
    };
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(transition_validation(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_transition_validation() {
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
    let d = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    transition_validation(&d).await;
}
