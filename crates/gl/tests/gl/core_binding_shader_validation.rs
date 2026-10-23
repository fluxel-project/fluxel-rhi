//! Core pipeline validation of shader resources against layout declarations.
use super::*;

async fn check(d: &Device<Gl>, source: &[u8], types: Vec<BindingType>) {
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = d.shader(shader("vs", ShaderStage::Vertex)).await.unwrap();
    let fragment = d.shader(shader("fs", ShaderStage::Fragment)).await.unwrap();
    let entries = types
        .into_iter()
        .enumerate()
        .map(|(binding, ty)| BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: vec![ShaderStage::Fragment],
            ty,
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        })
        .collect();
    let group = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries,
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group],
            push_constants: &[],
        })
        .unwrap();
    let result = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: vertex,
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![],
            color_targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: 15,
            })],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: u64::MAX,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
        .await;
    assert!(result.is_err(), "shader and layout type mismatch must fail");
}

async fn binding_shader_validation(d: &Device<Gl>) {
    let vertex = "@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));return vec4f(p[i],0.5,1.);}";
    let tex = |sample, dimension| BindingType::SampledTexture {
        sample,
        dimension,
        multisampled: false,
    };
    let uint = format!(
        "{vertex} @group(0) @binding(0) var t: texture_2d<u32>; @fragment fn fs()->@location(0) vec4f {{return vec4f(textureLoad(t,vec2i(0),0))/255.;}}"
    );
    check(
        d,
        uint.as_bytes(),
        vec![tex(TextureSampleType::Float, TextureViewDimension::D2)],
    )
    .await;
    let float = format!(
        "{vertex} @group(0) @binding(0) var t: texture_2d<f32>; @fragment fn fs()->@location(0) vec4f {{return textureLoad(t,vec2i(0),0);}}"
    );
    check(
        d,
        float.as_bytes(),
        vec![tex(TextureSampleType::Float, TextureViewDimension::D2Array)],
    )
    .await;
    let compare = format!(
        "{vertex} @group(0) @binding(0) var t: texture_depth_2d; @group(0) @binding(1) var s: sampler_comparison; @fragment fn fs()->@location(0) vec4f {{return vec4f(textureSampleCompare(t,s,vec2f(0.5),0.5));}}"
    );
    check(
        d,
        compare.as_bytes(),
        vec![
            tex(TextureSampleType::Depth, TextureViewDimension::D2),
            BindingType::Sampler {
                filtering: false,
                comparison: false,
            },
        ],
    )
    .await;
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_binding_shader_validation() {
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
    block_on(binding_shader_validation(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_binding_shader_validation() {
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
    block_on(binding_shader_validation(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_binding_shader_validation() {
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
    binding_shader_validation(&d).await;
}
