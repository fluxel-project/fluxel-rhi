//! Real single-lane execution of a reusable compiled RHI submission plan.
#![cfg(windows)]
use fluxel_rhi::api::*;
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

struct WakeThread(std::thread::Thread);
impl Wake for WakeThread {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
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
// On every failure, drain before the execution owners drop; failure to prove
// completion must not unwind through GPU-referenced resources.
struct Drain<B: CompletionBackend>(Device<B>);
impl<B: CompletionBackend> Drop for Drain<B> {
    fn drop(&mut self) {
        if let Err(e) = block_on(self.0.wait_idle()) {
            eprintln!("GPU drain failed: {e:?}");
            std::process::abort();
        }
    }
}
struct Execution<B: DeferredBackend + ReadbackBackend> {
    _buffer: Buffer<B>,
    ticket: Option<ReadbackTicket<B>>,
    work: Option<Vec<RecordedWork<B>>>,
    outcome: Option<SubmitOutcome<B>>,
    values: [u32; 3],
}
fn exercise<B: DeferredBackend + ReadbackBackend + ResourceBackend>(device: Device<B>) {
    let lane = device
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let id = PlanId {
        device: device.id(),
        ordinal: 987654,
    };
    let wait = |index| StepWait {
        previous: StepId { plan: id, index },
        destination: AccessScope {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ.union(AccessMask::COPY_WRITE),
        },
    };
    let steps = Arc::from([
        StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![],
        },
        StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![wait(0)],
        },
        StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![wait(0), wait(1)],
        },
        StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![wait(0), wait(1), wait(2)],
        },
    ]);
    // SAFETY: all steps are ordered on the same graphics lane, and explicit
    // waits identify preceding steps on this plan. Each run binds fresh work.
    let plan = unsafe { SubmissionPlan::<B>::from_compiled(id, steps) };
    let mut executions = Vec::new();
    let _drain = Drain(device.clone());
    for run in 0..4u32 {
        let buffer = device
            .create_buffer(&BufferDesc {
                label: Some("execution plan test".into()),
                size: 12,
                usage: BufferUsage::COPY_DST
                    .union(BufferUsage::COPY_SRC)
                    .union(BufferUsage::QUERY_RESOLVE),
                memory: MemoryClass::Device,
            })
            .unwrap();
        let values = [run * 10 + 1, run * 10 + 2, run * 10 + 3];
        let mut work = Vec::new();
        for (index, value) in values.iter().enumerate() {
            let mut encoder = device.encoder(lane).unwrap();
            encoder
                .write_buffer(&buffer, (index * 4) as u64, &value.to_le_bytes())
                .unwrap();
            work.push(encoder.finish().unwrap());
        }
        let mut encoder = device.encoder(lane).unwrap();
        let ticket = encoder.readback_buffer(&buffer, 0..12).unwrap();
        work.push(encoder.finish().unwrap());
        executions.push(Execution {
            _buffer: buffer,
            ticket: Some(ticket),
            work: Some(work),
            outcome: None,
            values,
        });
    }
    for (run, execution) in executions.iter_mut().enumerate() {
        let steps = execution
            .work
            .take()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, work)| BoundStep {
                step: plan.step(i),
                payload: StepPayload::Commands(vec![work]),
            })
            .collect();
        let completion = if run % 2 == 0 {
            CompletionRequest::End
        } else {
            CompletionRequest::Steps(Arc::from([plan.step(1), plan.step(3)]))
        };
        // SAFETY: independent buffers and tickets remain in `executions` until
        // Drain has observed idle, including all failed and uncertain outcomes.
        execution.outcome = Some(unsafe {
            device.queue().submit(SubmitPacket {
                plan: plan.clone(),
                steps,
                completion,
            })
        });
    }
    for (run, execution) in executions.iter_mut().enumerate() {
        let receipt = match execution.outcome.as_ref().unwrap() {
            SubmitOutcome::Accepted(receipt) => receipt,
            SubmitOutcome::Rejected { error, .. }
            | SubmitOutcome::PartiallyAccepted { error, .. }
            | SubmitOutcome::Indeterminate { error, .. } => panic!("submit {run}: {error:?}"),
        };
        assert_eq!(receipt.device, device.id());
        assert_eq!(
            receipt.accepted,
            (0..4).map(|i| plan.step(i)).collect::<Vec<_>>()
        );
        assert!(receipt.presents.is_empty());
        let expected = if run % 2 == 0 {
            vec![plan.step(3)]
        } else {
            vec![plan.step(1), plan.step(3)]
        };
        assert_eq!(receipt.checkpoints.len(), expected.len());
        for (checkpoint, frontier) in receipt.checkpoints.iter().zip(expected) {
            assert_eq!(checkpoint.point.device(), device.id());
            assert_eq!(checkpoint.covers.lane_frontiers, [(lane, frontier)]);
            block_on(device.wait_completion(&checkpoint.point)).unwrap();
        }
        let point = &receipt.checkpoints.last().unwrap().point;
        // SAFETY: the final checkpoint was observed complete and covers the
        // fourth step containing this execution's readback command.
        let data = block_on(unsafe { device.read_after(execution.ticket.take().unwrap(), point) })
            .unwrap();
        let expected: Vec<u8> = execution
            .values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(data.bytes, expected, "execution {run}");
    }
}
#[cfg(feature = "dx12")]
#[test]
#[ignore = "requires a Windows DX12 GPU"]
fn dx12_execution_plans() {
    use fluxel_rhi::dx12::Dx12Instance;
    exercise(block_on(Dx12Instance::new().unwrap().open_default()).unwrap());
}
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires a Windows Vulkan GPU"]
fn vulkan_execution_plans() {
    use fluxel_rhi::vulkan::VulkanInstance;
    exercise(block_on(VulkanInstance::new().unwrap().open_default()).unwrap());
}
