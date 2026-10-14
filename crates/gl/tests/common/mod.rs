//! Typed immediate-mode GL test workload helpers.
//!
//! Platform fixtures create the private GL context and then inject a typed
//! `Device<Gl>` plus its graphics lane. These helpers compile only the portable
//! immediate-session contract; they do not create contexts or replay commands.

use fluxel_rhi_core::api::*;
use std::sync::Arc;

/// Borrowed inputs for one immediate commands step.
pub(crate) struct ImmediateFixture<'a, B: ImmediateBackend> {
    /// The opened device whose context owns all native work.
    pub(crate) device: &'a Device<B>,
    /// The sole GL graphics lane selected from the opened capabilities.
    pub(crate) lane: LaneId,
}

/// Builds the one commands-step plan used by GL hardware fixtures.
///
/// The plan has no waits because GL begins with one ordered graphics lane.
/// A platform fixture must retain every object referenced by the subsequent
/// immediate encoder until the returned completion evidence has been observed.
pub(crate) fn single_commands_plan<B: ImmediateBackend>(
    fixture: &ImmediateFixture<'_, B>,
    ordinal: u64,
) -> Result<SubmissionPlan<B>> {
    if fixture.lane.device != fixture.device.id()
        || !fixture
            .device
            .lanes()
            .iter()
            .any(|lane| lane.id == fixture.lane)
    {
        return Err(Error::WrongDevice);
    }
    let id = PlanId {
        device: fixture.device.id(),
        ordinal,
    };
    // SAFETY: this helper creates one device-local commands step on the
    // fixture's discovered graphics lane and introduces no dependencies.
    Ok(unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([StepDesc {
                lane: fixture.lane,
                kind: StepKind::Commands,
                waits: Vec::new(),
            }]),
        )
    })
}

/// Starts an immediate GL commands step from a fixture-owned plan.
///
/// The caller records direct encoder verbs, closes the returned encoder, then
/// calls `ImmediateSession::finish` to obtain exactly the backend's completion
/// evidence. This helper intentionally does not hide close or finish outcomes.
///
/// # Safety
///
/// The fixture owner must keep every resource named by commands recorded into
/// the returned session alive until the session's issued prefix has completion
/// evidence. It must also close or abort the session explicitly so an
/// abandoned encoder cannot conceal native GL effects.
pub(crate) unsafe fn begin_commands<B: ImmediateBackend>(
    fixture: &ImmediateFixture<'_, B>,
    plan: SubmissionPlan<B>,
    completion: CompletionRequest,
) -> Result<ImmediateSession<B>> {
    // SAFETY: fixture ownership retains command-referenced resources until the
    // caller observes completion and explicitly finishes or aborts the session.
    unsafe { fixture.device.immediate(plan, completion) }
}

/// Obtains the drawable adopted with the typed GL device.
///
/// GL owns one adopted window drawable or canvas for the lifetime of a device.
/// The helper deliberately does not provide a pbuffer path: GL's public RHI
/// surface contract rejects it instead of pretending it is the adopted target.
pub(crate) fn adopted_surface(device: &Device<crate::Gl>) -> Result<Surface<crate::Gl>> {
    crate::Gl::surface(device)
}
