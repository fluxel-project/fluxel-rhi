//! Browser-worker SPIR-V translation regression coverage.

use super::*;
use std::{
    future::Future,
    task::{Context, Poll},
};
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::wasm_bindgen_test;

const COUNT_COMPUTE: &[u8] = include_bytes!("../../../tests/assets/worker_count.spv");

#[wasm_bindgen(module = "/src/backend/conformance/shader_worker.js")]
extern "C" {
    #[wasm_bindgen(js_name = nextTask)]
    fn next_task() -> js_sys::Promise;
}

fn shader_desc(bytes: &[u8]) -> ShaderDesc {
    ShaderDesc {
        label: Some("worker-spv".into()),
        format: ShaderFormat::SpirV,
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    }
}

fn configure_worker() {
    let url = option_env!("FLUXEL_WEBGPU_TEST_SHADER_WORKER_URL")
        .expect("browser harness must supply FLUXEL_WEBGPU_TEST_SHADER_WORKER_URL");
    WebGpuInstance::new()
        .unwrap()
        .configure_spirv_worker(url)
        .unwrap();
}

#[wasm_bindgen_test(async)]
async fn spirv_is_unsupported_when_worker_is_not_configured() {
    // The full browser suite also tests configured-worker paths in this JS
    // realm, so restore the unconfigured deployment state explicitly.
    super::super::shader_worker::reset_configuration_for_tests();
    let instance = WebGpuInstance::new().unwrap();
    let device = instance.open_default().await.unwrap();
    let desc = shader_desc(COUNT_COMPUTE);
    assert!(matches!(
        device.shader_route(&desc),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        device.shader(desc).await,
        Err(Error::Unsupported("shader format/profile/ABI"))
    ));
}

#[wasm_bindgen_test(async)]
async fn worker_spirv_shader_builds_pipeline_and_dispatches() {
    configure_worker();
    let d = open().await;
    let shader = d.shader(shader_desc(COUNT_COMPUTE)).await.unwrap();
    let output = buffer(
        &d,
        80,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    let group_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 80,
                    access: StorageAccess::ReadWrite,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &output,
                    range: 0..80,
                },
            }],
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let compute = d.compute().unwrap();
    let pipeline = compute
        .create_pipeline(ComputePipelineDesc {
            label: None,
            layout,
            compute: PipelineStage {
                module: shader,
                entry: "main".into(),
                constants: vec![],
            },
            required_subgroup_size: None,
            cache: None,
        })
        .await
        .unwrap();
    let mut encoder = d.encoder(d.lanes()[0].id).unwrap();
    let mut pass = compute.begin(&mut encoder).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.finish().unwrap();
    let ticket = encoder.readback_buffer(&output, 64..80).unwrap();
    let point = submit(&d, encoder).await;
    assert_eq!(
        unsafe { d.read_after(ticket, &point).await }.unwrap().bytes,
        [0u32, 1, 2, 9]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
}

#[wasm_bindgen_test(async)]
async fn worker_error_and_cancellation_do_not_strand_later_translation() {
    configure_worker();
    assert!(
        super::super::shader_worker::translate_async(vec![0, 1, 2, 3])
            .await
            .is_err()
    );

    let mut cancelled = Box::pin(super::super::shader_worker::translate_async(
        COUNT_COMPUTE.to_vec(),
    ));
    let mut context = Context::from_waker(std::task::Waker::noop());
    assert!(matches!(
        cancelled.as_mut().poll(&mut context),
        Poll::Pending
    ));
    drop(cancelled);
    assert!(
        super::super::shader_worker::translate_async(COUNT_COMPUTE.to_vec())
            .await
            .is_ok()
    );
}

#[wasm_bindgen_test(async)]
async fn worker_admission_returns_before_a_browser_task() {
    configure_worker();
    super::super::shader_worker::hold_next_for_test();
    let mut future = Box::pin(super::super::shader_worker::translate_async(
        COUNT_COMPUTE.to_vec(),
    ));
    // If translation ran in this call on the owner thread, the zero-delay task
    // could not be scheduled until Naga had already completed.
    JsFuture::from(next_task())
        .await
        .map_err(|error: JsValue| format!("{error:?}"))
        .unwrap();
    let mut context = Context::from_waker(std::task::Waker::noop());
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    super::super::shader_worker::release_hold_for_test();
    assert!(future.await.is_ok());
}
