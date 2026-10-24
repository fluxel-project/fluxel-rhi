//! Browser GPU output assertions for the completeness implementation.
use super::{WebGpu, WebGpuInstance};
use fluxel_rhi_core::api::*;
use std::sync::Arc;
use wasm_bindgen_test::*;
#[path = "conformance/binding_validation.rs"]
mod binding_validation;
#[path = "conformance/blit_usage.rs"]
mod blit_usage;
#[path = "conformance/buffer_usage.rs"]
mod buffer_usage;
#[path = "conformance/buffers.rs"]
mod buffers;
#[path = "conformance/completion_steps.rs"]
mod completion_steps;
#[path = "conformance/multisample.rs"]
mod multisample;
#[path = "conformance/operations.rs"]
mod operations;
#[path = "conformance/queries.rs"]
mod queries;
#[path = "conformance/query_capacity.rs"]
mod query_capacity;
#[path = "conformance/raster.rs"]
mod raster;
#[path = "conformance/resolve.rs"]
mod resolve;
#[path = "conformance/shader.rs"]
mod shader;
#[path = "conformance/storage_view.rs"]
mod storage_view;
#[path = "conformance/surface.rs"]
mod surface;
#[path = "conformance/textures.rs"]
mod textures;
#[path = "conformance/utility_cache.rs"]
mod utility_cache;

#[wasm_bindgen_test(async)]
async fn browser_gpu_environment_records_adapter_and_features() {
    let gpu = super::js::browser_gpu().unwrap();
    let adapter =
        wasm_bindgen_futures::JsFuture::from(super::js::request_adapter(&gpu, None).unwrap())
            .await
            .unwrap();
    let info = super::js::property(&adapter, "info").unwrap();
    let output: wasm_bindgen::JsValue = js_sys::Object::new().into();
    for key in [
        "vendor",
        "architecture",
        "device",
        "description",
        "isFallbackAdapter",
    ] {
        if let Ok(value) = super::js::property(&info, key) {
            super::js::set(&output, key, &value).unwrap();
        }
    }
    let features = super::js::property(&adapter, "features").unwrap();
    super::js::set(&output, "features", &js_sys::Array::from(&features).into()).unwrap();
    let navigator = super::js::property(&js_sys::global().into(), "navigator").unwrap();
    super::js::set(
        &output,
        "userAgent",
        &super::js::property(&navigator, "userAgent").unwrap(),
    )
    .unwrap();
    console_log!(
        "WEBGPU_ENV {}",
        js_sys::JSON::stringify(&output)
            .unwrap()
            .as_string()
            .unwrap()
    );
}

async fn open() -> Device<WebGpu> {
    WebGpuInstance::new().unwrap().open_default().await.unwrap()
}

async fn submit(d: &Device<WebGpu>, encoder: CommandEncoder<WebGpu>) -> CompletionPoint<WebGpu> {
    let lane = d.lanes()[0].id;
    // SAFETY: this fixture owns all resources and submits one ordered recording.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 120021,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the recording belongs to this plan's only lane and has no hazards with other work.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![encoder.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("browser rejected submission"),
    };
    d.wait_completion(&point).await.unwrap();
    point
}

fn buffer(
    d: &Device<WebGpu>,
    size: u64,
    usage: BufferUsage,
    memory: MemoryClass,
) -> Buffer<WebGpu> {
    d.create_buffer(&BufferDesc {
        label: None,
        size,
        usage,
        memory,
    })
    .unwrap()
}
