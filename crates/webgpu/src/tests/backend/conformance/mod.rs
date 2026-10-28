//! Browser GPU output assertions for the completeness implementation.
use super::{WebGpu, WebGpuInstance};
use fluxel_rhi_core::api::*;
use std::sync::Arc;
use wasm_bindgen_test::*;
#[path = "binding_validation.rs"]
mod binding_validation;
#[path = "blit_usage.rs"]
mod blit_usage;
#[path = "buffer_usage.rs"]
mod buffer_usage;
#[path = "buffers.rs"]
mod buffers;
#[path = "completion_steps.rs"]
mod completion_steps;
#[path = "multisample.rs"]
mod multisample;
#[path = "operations.rs"]
mod operations;
#[path = "pass_split.rs"]
mod pass_split;
#[path = "queries.rs"]
mod queries;
#[path = "query_capacity.rs"]
mod query_capacity;
#[path = "raster.rs"]
mod raster;
#[path = "resolve.rs"]
mod resolve;
#[path = "shader.rs"]
mod shader;
#[path = "storage_view.rs"]
mod storage_view;
#[path = "surface.rs"]
mod surface;
#[path = "textures.rs"]
mod textures;
#[path = "transient.rs"]
mod transient;
#[path = "utility_cache.rs"]
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
    let instance = WebGpuInstance::new().unwrap();
    if let Some(url) = option_env!("FLUXEL_WEBGPU_TEST_SHADER_WORKER_URL") {
        instance.configure_spirv_worker(url).unwrap();
    }
    instance.open_default().await.unwrap()
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

#[path = "completion_retirement.rs"]
mod completion_retirement;
#[path = "indirect_count_packing.rs"]
mod indirect_count_packing;
#[path = "stencil_copy.rs"]
mod stencil_copy;
#[path = "upload_arena.rs"]
mod upload_arena;

#[path = "shader_worker.rs"]
mod shader_worker;
