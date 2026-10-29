//! Headless 18 ToyRobot mesh scene on Vulkan, using Vulkan SPIR-V variants.
#![cfg(all(windows, feature = "hardware-tests"))]
#[path = "support/mod.rs"]
mod support;
use fluxel_rhi_core::api::*;
use fluxel_rhi_core::backend::{
    DeferredBackend, PipelineBackend, RasterBackend, ReadbackBackend, ResourceBackend,
    ShaderBackend,
};
use fluxel_rhi_vulkan::VulkanInstance;
use std::sync::Arc;
use support::block_on;
#[path = "../../rhi/examples/18_meshlet_instancing/model.rs"]
mod model;
#[path = "toy/record.rs"]
mod record;
#[path = "../../rhi/examples/18_meshlet_instancing/scene.rs"]
mod scene;
#[path = "toy/setup.rs"]
mod setup;
struct Demo<B: ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + ReadbackBackend>
{
    lane: LaneId,
    buffers: Vec<Buffer<B>>,
    output: Texture<B>,
    view: TextureView<B>,
    depth: Texture<B>,
    depth_view: TextureView<B>,
    readback: Buffer<B>,
    row_pitch: u32,
    group: BindGroup<B>,
    pipeline: MeshPipeline<B>,
    initialized: bool,
}
#[test]
fn toy_robot_readback() {
    let d = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    if !d.capabilities().mesh {
        eprintln!("SKIP mesh unsupported");
        return;
    }
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let extent = [1280, 720];
    let demo = block_on(Demo::new(&d, lane, extent, TextureFormat::Rgba8Unorm)).unwrap();
    let (work, ticket) = demo.record(&d, extent).unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 1800001,
    };
    // SAFETY: one recorded work belongs to this device and lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: work and plan share device and lane, and work is consumed once.
    let receipt = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match receipt {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submit failed"),
    };
    block_on(d.wait_completion(&point)).unwrap();
    // SAFETY: completion covers the readback command.
    let bytes = unsafe {
        block_on(d.read_after(ticket.unwrap(), &point))
            .unwrap()
            .bytes
    };
    record::verify_and_save(&demo, extent, &bytes).unwrap();
    let foreground = bytes.chunks_exact(4).filter(|p| p[0] > 8).count();
    assert!(foreground > 10000, "foreground {foreground}");
}
