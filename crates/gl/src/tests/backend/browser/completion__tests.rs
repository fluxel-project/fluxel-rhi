use super::*;
use js_sys::{Function, Object, Reflect};
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::WebGl2RenderingContext as Gl;

#[wasm_bindgen_test]
fn headless_checkpoints_publish_once_on_observation() {
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    let discovery = crate::backend::browser::WebGl2BrowserDiscovery::open(
        crate::next_context_stamp().unwrap(),
        canvas,
    )
    .unwrap();
    let owner = BrowserOwner::new(discovery);
    let raw = owner.state.borrow().context.raw.clone();
    let counts = Object::new();
    for name in ["flush", "error"] {
        Reflect::set(&counts, &JsValue::from_str(name), &JsValue::from_f64(0.0)).unwrap();
    }
    Reflect::set(raw.as_ref(), &JsValue::from_str("__fluxel_counts"), &counts).unwrap();
    for (method, counter) in [("flush", "flush"), ("getError", "error")] {
        let original = Reflect::get(raw.as_ref(), &JsValue::from_str(method)).unwrap();
        Reflect::set(
            raw.as_ref(),
            &JsValue::from_str(&format!("__fluxel_original_{method}")),
            &original,
        )
        .unwrap();
        let wrapper = Function::new_no_args(&format!(
            "const c=this.__fluxel_counts;c.{counter}++;return this.__fluxel_original_{method}.apply(this,arguments);"
        ));
        Reflect::set(raw.as_ref(), &JsValue::from_str(method), wrapper.as_ref()).unwrap();
    }
    let first = owner.completion_after_current_commands().unwrap();
    let second = owner.completion_after_current_commands().unwrap();
    assert_eq!(count(&counts, "flush"), 0.0);
    assert_eq!(count(&counts, "error"), 0.0);
    let _ = owner.completion_status(&first);
    assert_eq!(count(&counts, "flush"), 1.0);
    assert_eq!(count(&counts, "error"), 1.0);
    let _ = owner.completion_status(&second);
    let _ = owner.completion_status(&first);
    assert_eq!(count(&counts, "flush"), 1.0);
    assert_eq!(count(&counts, "error"), 1.0);
}

#[wasm_bindgen_test]
fn wait_failure_fails_every_owner_completion() {
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    let discovery = crate::backend::browser::WebGl2BrowserDiscovery::open(
        crate::next_context_stamp().unwrap(),
        canvas,
    )
    .unwrap();
    let owner = BrowserOwner::new(discovery);
    let first = owner.completion_after_current_commands().unwrap();
    let second = owner.completion_after_current_commands().unwrap();
    let raw = owner.state.borrow().context.raw.clone();
    let failed = Function::new_no_args("return 0x911d;"); // WAIT_FAILED
    Reflect::set(
        raw.as_ref(),
        &JsValue::from_str("clientWaitSync"),
        failed.as_ref(),
    )
    .unwrap();
    assert!(matches!(
        owner.completion_status(&first),
        CompletionState::Failed(Error::Backend(_))
    ));
    assert!(matches!(
        owner.completion_status(&second),
        CompletionState::Failed(Error::Backend(_))
    ));
}

#[wasm_bindgen_test(async)]
async fn context_loss_fails_every_pending_completion_permanently() {
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    let discovery = crate::backend::browser::WebGl2BrowserDiscovery::open(
        crate::next_context_stamp().unwrap(),
        canvas,
    )
    .unwrap();
    let owner = BrowserOwner::new(discovery);
    let first = owner.completion_after_current_commands().unwrap();
    let second = owner.completion_after_current_commands().unwrap();
    let raw = owner.state.borrow().context.raw.clone();
    let extension = raw
        .get_extension("WEBGL_lose_context")
        .unwrap()
        .expect("browser WebGL context-loss extension");
    let lose = Reflect::get(&extension, &JsValue::from_str("loseContext"))
        .unwrap()
        .dyn_into::<Function>()
        .unwrap();
    lose.call0(&extension).unwrap();
    for _ in 0..5 {
        if raw.is_context_lost() {
            break;
        }
        next_browser_task().await;
    }
    assert!(raw.is_context_lost());
    for completion in [&first, &second, &first] {
        assert!(matches!(
            owner.completion_status(completion),
            CompletionState::Failed(Error::DeviceLost)
        ));
    }
}

#[wasm_bindgen_test]
fn warmed_raster_frames_do_not_query_state_or_program_metadata() {
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    let discovery = crate::backend::browser::WebGl2BrowserDiscovery::open(
        crate::next_context_stamp().unwrap(),
        canvas,
    )
    .unwrap();
    let owner = BrowserOwner::new(discovery);
    owner.resize_drawing_buffer([4, 4]).unwrap();
    let raw = owner.state.borrow().context.raw.clone();
    let program = program(&raw);
    let vao = raw.create_vertex_array().unwrap();

    // Compile/link and VAO allocation are deliberately outside the measured
    // warm path. The counter sees only three identical direct raster frames.
    let counts = install_counters(
        &raw,
        [
            ("getParameter", "parameter"),
            ("getUniformLocation", "uniform"),
            ("getProgramParameter", "program"),
            ("checkFramebufferStatus", "framebuffer"),
            ("getError", "error"),
            ("flush", "flush"),
            ("finish", "finish"),
        ],
    );
    for _ in 0..3 {
        let before = counters(&counts);
        let mut encoder = owner
            .begin_encoder(fluxel_rhi_core::api::LaneId {
                device: fluxel_rhi_core::api::DeviceId::allocate().unwrap(),
                ordinal: 0,
            })
            .unwrap();
        encoder
            .begin_default_raster(
                [4, 4],
                Some(fluxel_rhi_core::api::ClearColor::Float([
                    0.0, 0.0, 0.0, 1.0,
                ])),
                fluxel_rhi_core::api::Rect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
            )
            .unwrap();
        encoder.set_program(&program).unwrap();
        encoder.set_vertex_array(&vao).unwrap();
        encoder.draw_arrays(Gl::TRIANGLES, 0, 3, 1).unwrap();
        encoder.end_raster().unwrap();
        encoder.close_scope().unwrap();
        owner.flush().unwrap();
        let after = counters(&counts);
        assert_eq!(after.parameter - before.parameter, 0.0);
        assert_eq!(after.uniform - before.uniform, 0.0);
        assert_eq!(after.program - before.program, 0.0);
        assert_eq!(after.framebuffer - before.framebuffer, 0.0);
        assert!(after.error - before.error <= 1.0);
        assert!(after.flush - before.flush <= 1.0);
        assert_eq!(after.finish - before.finish, 0.0);
    }

    // Pixel validation intentionally follows the counter assertions. It is an
    // explicit result read and therefore not part of the normal-frame budget.
    let mut pixel = [0; 4];
    raw.read_pixels_with_opt_u8_array(2, 2, 1, 1, Gl::RGBA, Gl::UNSIGNED_BYTE, Some(&mut pixel))
        .unwrap();
    assert_eq!(pixel, [0, 255, 0, 255]);
}

fn program(raw: &Gl) -> web_sys::WebGlProgram {
    let compile = |kind, source: &str| {
        let shader = raw.create_shader(kind).unwrap();
        raw.shader_source(&shader, source);
        raw.compile_shader(&shader);
        assert_eq!(
            raw.get_shader_parameter(&shader, Gl::COMPILE_STATUS)
                .as_bool(),
            Some(true)
        );
        shader
    };
    let vertex = compile(
        Gl::VERTEX_SHADER,
        "#version 300 es\nvoid main(){vec2 p[3]=vec2[3](vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));gl_Position=vec4(p[gl_VertexID],0.,1.);}",
    );
    let fragment = compile(
        Gl::FRAGMENT_SHADER,
        "#version 300 es\nprecision highp float;out vec4 color;void main(){color=vec4(0.,1.,0.,1.);}",
    );
    let program = raw.create_program().unwrap();
    raw.attach_shader(&program, &vertex);
    raw.attach_shader(&program, &fragment);
    raw.link_program(&program);
    raw.delete_shader(Some(&vertex));
    raw.delete_shader(Some(&fragment));
    assert_eq!(
        raw.get_program_parameter(&program, Gl::LINK_STATUS)
            .as_bool(),
        Some(true)
    );
    program
}

#[derive(Clone, Copy)]
struct Counters {
    parameter: f64,
    uniform: f64,
    program: f64,
    framebuffer: f64,
    error: f64,
    flush: f64,
    finish: f64,
}

fn install_counters<const N: usize>(raw: &Gl, methods: [(&str, &str); N]) -> Object {
    let counts = Object::new();
    for (_, counter) in methods {
        Reflect::set(
            &counts,
            &JsValue::from_str(counter),
            &JsValue::from_f64(0.0),
        )
        .unwrap();
    }
    Reflect::set(raw.as_ref(), &JsValue::from_str("__fluxel_counts"), &counts).unwrap();
    for (method, counter) in methods {
        let original = Reflect::get(raw.as_ref(), &JsValue::from_str(method)).unwrap();
        Reflect::set(
            raw.as_ref(),
            &JsValue::from_str(&format!("__fluxel_original_{method}")),
            &original,
        )
        .unwrap();
        let wrapper = Function::new_no_args(&format!(
            "const c=this.__fluxel_counts;c.{counter}++;return this.__fluxel_original_{method}.apply(this,arguments);"
        ));
        Reflect::set(raw.as_ref(), &JsValue::from_str(method), wrapper.as_ref()).unwrap();
    }
    counts
}

fn counters(counts: &Object) -> Counters {
    Counters {
        parameter: count(counts, "parameter"),
        uniform: count(counts, "uniform"),
        program: count(counts, "program"),
        framebuffer: count(counts, "framebuffer"),
        error: count(counts, "error"),
        flush: count(counts, "flush"),
        finish: count(counts, "finish"),
    }
}

async fn next_browser_task() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let resolve = resolve.clone();
        let callback = Closure::once_into_js(move || {
            resolve.call0(&JsValue::NULL).unwrap();
        });
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(callback.unchecked_ref(), 0)
            .unwrap();
    });
    wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
}

fn count(object: &Object, name: &str) -> f64 {
    Reflect::get(object, &JsValue::from_str(name))
        .unwrap()
        .as_f64()
        .unwrap()
}
