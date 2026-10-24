//! Queue-promise retirement without subsequent RHI operations.
use super::*;
use js_sys::Promise;
use std::rc::Rc;
use wasm_bindgen_futures::JsFuture;

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
