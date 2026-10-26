use std::cell::Cell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FailureStage {
    Event,
    Wait,
    SetEvent,
}

thread_local! {
    static FAILURE_STAGE: Cell<Option<FailureStage>> = const { Cell::new(None) };
}

pub(super) fn fail_here(stage: FailureStage) -> bool {
    FAILURE_STAGE.with(|slot| {
        if slot.get() == Some(stage) {
            slot.set(None);
            true
        } else {
            false
        }
    })
}

#[cfg(feature = "hardware-tests")]
pub(super) fn inject_failure(stage: FailureStage) {
    FAILURE_STAGE.with(|slot| slot.set(Some(stage)));
}
