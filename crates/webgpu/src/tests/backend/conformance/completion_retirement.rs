//! Queue-promise retirement without subsequent RHI operations.
use super::*;
use fluxel_rhi_core::backend::CompletionBackend;
use js_sys::Promise;
use std::rc::Rc;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

fn retain_uncertain(native: &super::super::instance::WebGpuDevice) {
    let sequence = native.0.next_uncertain_resource.get();
    native.0.next_uncertain_resource.set(sequence + 1);
    native
        .0
        .uncertain_resources
        .borrow_mut()
        .push(super::super::instance::UncertainResource {
            sequence,
            _value: JsValue::UNDEFINED,
        });
}

#[wasm_bindgen_test(async)]
async fn idle_releases_only_uncertain_resources_before_its_queue_boundary() {
    let d = open().await;
    // SAFETY: private immutable bookkeeping access for this browser fixture.
    let native = unsafe { d.native_ref() }.clone();
    retain_uncertain(&native); // before both idle boundaries
    let first = <WebGpu as CompletionBackend>::wait_idle(&native);
    retain_uncertain(&native); // after first, before second boundary
    let second = <WebGpu as CompletionBackend>::wait_idle(&native);
    retain_uncertain(&native); // after both boundaries

    first.await.unwrap();
    assert_eq!(
        native
            .0
            .uncertain_resources
            .borrow()
            .iter()
            .map(|entry| entry.sequence)
            .collect::<Vec<_>>(),
        [1, 2]
    );
    second.await.unwrap();
    assert_eq!(
        native
            .0
            .uncertain_resources
            .borrow()
            .iter()
            .map(|entry| entry.sequence)
            .collect::<Vec<_>>(),
        [2]
    );
}

#[wasm_bindgen_test(async)]
async fn failed_or_dropped_idle_keeps_uncertain_resources() {
    let d = open().await;
    // SAFETY: private immutable bookkeeping access for this browser fixture.
    let native = unsafe { d.native_ref() }.clone();
    retain_uncertain(&native);
    let cancelled = <WebGpu as CompletionBackend>::wait_idle(&native);
    drop(cancelled);
    assert_eq!(native.0.uncertain_resources.borrow().len(), 1);

    let failed = <WebGpu as CompletionBackend>::wait_idle(&native);
    *native.0.validation_error.borrow_mut() = Some("injected idle failure".into());
    assert!(failed.await.is_err());
    assert_eq!(native.0.uncertain_resources.borrow().len(), 1);
}

#[wasm_bindgen_test(async)]
async fn gpu_none_submission_retires_without_followup_rhi_calls() {
    let d = open().await;
    let lane = d.lanes()[0].id;
    let b = buffer(
        &d,
        16,
        BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    let mut e = d.encoder(lane).unwrap();
    e.write_buffer(&b, 0, &[7u8; 16]).unwrap();
    // SAFETY: the device, lane and single recording all have matching identity.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 120030,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan: plan.clone(),
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            }],
            completion: CompletionRequest::None,
        })
    };
    assert!(matches!(outcome, SubmitOutcome::Accepted(_)));
    // SAFETY: private immutable bookkeeping access, no queue retirement API.
    let native = unsafe { d.native_ref() }.clone();
    assert_eq!(native.0.in_flight.borrow().len(), 1);
    let weak = Rc::downgrade(&native.0);
    let done = super::super::js::call0(&native.0.queue, "onSubmittedWorkDone").unwrap();
    drop(outcome);
    drop(plan);
    drop(b);
    drop(d);
    drop(native);
    // Wait only on the native browser queue, then drain already-scheduled
    // microtasks. No RHI query/submit/wait can incidentally retire resources.
    JsFuture::from(Promise::from(done)).await.unwrap();
    for _ in 0..4 {
        JsFuture::from(Promise::resolve(&wasm_bindgen::JsValue::UNDEFINED))
            .await
            .unwrap();
    }
    assert!(
        weak.upgrade().is_none(),
        "promise task must retire and release the idle Rust device owner"
    );
}
