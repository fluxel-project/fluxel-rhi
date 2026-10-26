//! Browser WebGPU completion evidence at multiple native queue boundaries.

use super::*;

#[wasm_bindgen_test(async)]
async fn gpu_steps_publish_each_requested_queue_boundary() {
    let d = open().await;
    let lane = d.lanes()[0].id;
    let target = buffer(
        &d,
        12,
        BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    let mut first = d.encoder(lane).unwrap();
    first
        .write_buffer(&target, 0, &11u32.to_le_bytes())
        .unwrap();
    let mut second = d.encoder(lane).unwrap();
    second
        .write_buffer(&target, 4, &22u32.to_le_bytes())
        .unwrap();
    let mut third = d.encoder(lane).unwrap();
    third
        .write_buffer(&target, 8, &33u32.to_le_bytes())
        .unwrap();
    let ticket = third.readback_buffer(&target, 0..12).unwrap();

    // SAFETY: the three records and one lane all originate from `d`; each
    // step receives exactly its matching command payload below.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 120022,
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
    // Requests deliberately differ from execution order and repeat the last
    // boundary. The receipt must preserve this public request order while the
    // backend emits each native boundary only once.
    let requested = [plan.step(2), plan.step(0), plan.step(2)];
    // SAFETY: the plan, lane, and all three one-shot records have the same
    // device identity; the fixture consumes every record exactly once.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan: plan.clone(),
            steps: vec![
                BoundStep {
                    step: plan.step(0),
                    payload: StepPayload::Commands(vec![first.finish().unwrap()]),
                },
                BoundStep {
                    step: plan.step(1),
                    payload: StepPayload::Commands(vec![second.finish().unwrap()]),
                },
                BoundStep {
                    step: plan.step(2),
                    payload: StepPayload::Commands(vec![third.finish().unwrap()]),
                },
            ],
            completion: CompletionRequest::Steps(Arc::from(requested)),
        })
    };
    let receipt = match outcome {
        SubmitOutcome::Accepted(receipt) => receipt,
        SubmitOutcome::Rejected { error, .. }
        | SubmitOutcome::PartiallyAccepted { error, .. }
        | SubmitOutcome::Indeterminate { error, .. } => {
            panic!("WebGPU steps submit failed: {error:?}")
        }
    };
    assert_eq!(receipt.accepted, [plan.step(0), plan.step(1), plan.step(2)]);
    assert_eq!(receipt.checkpoints.len(), requested.len());
    for (checkpoint, step) in receipt.checkpoints.iter().zip(requested) {
        assert_eq!(checkpoint.covers.lane_frontiers, [(lane, step)]);
        d.wait_completion(&checkpoint.point).await.unwrap();
    }
    // SAFETY: the final requested point is a real queue boundary after the
    // third encoder's write and readback copy.
    assert_eq!(
        unsafe { d.read_after(ticket, &receipt.checkpoints[0].point).await }
            .unwrap()
            .bytes,
        [11u32, 22, 33]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
}
