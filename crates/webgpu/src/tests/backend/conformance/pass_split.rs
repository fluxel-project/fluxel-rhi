//! Annotation state is preserved when an active raster pass is reopened.
use super::*;
use js_sys::{Array, Object, Reflect};
use std::{cell::RefCell, rc::Rc};
use wasm_bindgen::{JsValue, closure::Closure};

fn set(object: &Object, name: &str, value: &JsValue) {
    Reflect::set(object, &JsValue::from_str(name), value).unwrap();
}

fn descriptor(colors: &[(bool, &str)], depth: Option<(&str, &str, bool, bool)>) -> JsValue {
    let descriptor = Object::new();
    let attachments = Array::new();
    for &(present, store) in colors {
        if !present {
            attachments.push(&JsValue::NULL);
            continue;
        }
        let attachment = Object::new();
        set(&attachment, "storeOp", &JsValue::from_str(store));
        attachments.push(&attachment);
    }
    set(&descriptor, "colorAttachments", attachments.as_ref());
    set(&descriptor, "depthStencilAttachment", &JsValue::NULL);
    if let Some((depth_store, stencil_store, depth_read_only, stencil_read_only)) = depth {
        let attachment = Object::new();
        set(&attachment, "depthStoreOp", &JsValue::from_str(depth_store));
        set(
            &attachment,
            "stencilStoreOp",
            &JsValue::from_str(stencil_store),
        );
        set(
            &attachment,
            "depthReadOnly",
            &JsValue::from_bool(depth_read_only),
        );
        set(
            &attachment,
            "stencilReadOnly",
            &JsValue::from_bool(stencil_read_only),
        );
        set(&descriptor, "depthStencilAttachment", attachment.as_ref());
    }
    descriptor.into()
}

fn state(descriptor: JsValue, dummy: bool) -> JsValue {
    let state = Object::new();
    set(&state, "descriptor", &descriptor);
    set(&state, "dummy", &JsValue::from_bool(dummy));
    state.into()
}

#[wasm_bindgen_test]
fn raster_split_rejects_discarded_writable_attachments() {
    for descriptor in [
        descriptor(&[(true, "discard")], None),
        descriptor(&[], Some(("discard", "store", false, false))),
        descriptor(&[], Some(("store", "discard", false, false))),
    ] {
        assert!(matches!(
            crate::backend::pass_split::validate_split_stores(&state(descriptor, false)),
            Err(Error::Unsupported(
                "WebGPU internal pass split with discarded attachments"
            ))
        ));
    }
}

#[wasm_bindgen_test]
fn raster_split_allows_stored_attachments_and_read_only_aspects() {
    for descriptor in [
        descriptor(&[(true, "store")], None),
        descriptor(&[], Some(("discard", "discard", true, true))),
        descriptor(
            &[(false, "discard")],
            Some(("store", "store", false, false)),
        ),
    ] {
        crate::backend::pass_split::validate_split_stores(&state(descriptor, false)).unwrap();
    }
}

#[wasm_bindgen_test]
fn raster_split_allows_the_backend_owned_dummy_discard_attachment() {
    let descriptor = descriptor(&[], Some(("discard", "discard", false, false)));
    crate::backend::pass_split::validate_split_stores(&state(descriptor, true)).unwrap();
}

#[wasm_bindgen_test(async)]
async fn gpu_empty_raster_pass_splits_to_four_sample_pipeline() {
    let d = open().await;
    let dummy_desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 4,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    };
    if !matches!(d.texture_support(&dummy_desc), Support::Supported(_)) {
        console_log!("SKIP multisampled depth32float dummy attachment");
        return;
    }

    let shader_source = "@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.); }";
    let shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(shader_source.as_bytes()),
            entries: vec![ShaderEntryPoint {
                name: "vs".into(),
                stage: ShaderStage::Vertex,
                workgroup_size: None,
            }],
            required_features: vec![],
        })
        .await
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let face = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: shader,
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: None,
            vertex_buffers: vec![],
            color_targets: vec![],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: CompareFunction::Always,
                stencil_front: face,
                stencil_back: face,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                depth_bias_constant: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            multisample: MultisampleState {
                count: 4,
                mask: u64::MAX,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();

    let mut encoder = d.encoder(d.lanes()[0].id).unwrap();
    let mut pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    submit(&d, encoder).await;
}

#[wasm_bindgen_test]
fn raster_split_restores_nested_debug_groups_in_order() {
    let calls = Rc::new(RefCell::new(Vec::<String>::new()));
    let observed = calls.clone();
    let push = Closure::<dyn FnMut(JsValue)>::new(move |label: JsValue| {
        observed
            .borrow_mut()
            .push(label.as_string().expect("debug group label"));
    });
    let receiver = Object::new();
    Reflect::set(
        &receiver,
        &JsValue::from_str("pushDebugGroup"),
        push.as_ref(),
    )
    .unwrap();
    let labels = vec![
        "render graph pass".to_owned(),
        "nested operation".to_owned(),
    ];

    crate::backend::command::restore_debug_groups(&labels, receiver.as_ref()).unwrap();

    assert_eq!(&*calls.borrow(), &["render graph pass", "nested operation"]);
}
