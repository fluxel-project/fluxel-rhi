//! TLAS dependency snapshots and their native-completion publication.
//!
//! This state machine owns no GPU scheduling. A recording sees its own
//! replacement; completed work publishes that replacement only after Metal
//! reports the command buffer terminal, and the publication completion lets
//! the submit path make this ordering observable to Core.

use super::MetalAcceleration;
use fluxel_rhi_core::api::{self, Error};
use std::sync::{Arc, Mutex};

pub(crate) struct FrozenDependencies<T> {
    phase: Arc<Mutex<BuildState>>,
    values: Mutex<Vec<T>>,
    #[cfg(target_vendor = "apple")]
    contributions: Mutex<Vec<u32>>,
    recording: Mutex<Option<Recording<T>>>,
}
struct Recording<T> {
    owner: u64,
    values: Vec<T>,
    #[cfg(target_vendor = "apple")]
    contributions: Vec<u32>,
}
enum BuildState {
    Empty,
    Recording(u64, bool),
    Submitted,
    Ready,
    Failed,
}

impl<T> Default for FrozenDependencies<T> {
    fn default() -> Self {
        Self {
            phase: Arc::new(Mutex::new(BuildState::Empty)),
            values: Mutex::new(Vec::new()),
            #[cfg(target_vendor = "apple")]
            contributions: Mutex::new(Vec::new()),
            recording: Mutex::new(None),
        }
    }
}
impl<T> FrozenDependencies<T> {
    pub(super) fn with_snapshot<R>(&self, owner: u64, f: impl FnOnce(&[T]) -> R) -> api::Result<R> {
        let state = self.phase.lock().unwrap_or_else(|p| p.into_inner());
        match &*state {
            BuildState::Recording(current, _) if *current == owner => {
                let recording = self.recording.lock().unwrap_or_else(|p| p.into_inner());
                Ok(f(&recording.as_ref().ok_or(Error::InvalidState)?.values))
            }
            BuildState::Ready => Ok(f(&self.values.lock().unwrap_or_else(|p| p.into_inner()))),
            _ => Err(Error::InvalidState),
        }
    }
    #[cfg(target_vendor = "apple")]
    pub(super) fn with_contributions<R>(
        &self,
        owner: u64,
        f: impl FnOnce(&[u32]) -> R,
    ) -> api::Result<R> {
        let state = self.phase.lock().unwrap_or_else(|p| p.into_inner());
        match &*state {
            BuildState::Recording(current, _) if *current == owner => {
                let recording = self.recording.lock().unwrap_or_else(|p| p.into_inner());
                Ok(f(&recording
                    .as_ref()
                    .ok_or(Error::InvalidState)?
                    .contributions))
            }
            BuildState::Ready => Ok(f(&self
                .contributions
                .lock()
                .unwrap_or_else(|p| p.into_inner()))),
            _ => Err(Error::InvalidState),
        }
    }
    pub(super) fn freeze(
        &self,
        owner: u64,
        values: Vec<T>,
        #[cfg(target_vendor = "apple")] contributions: Vec<u32>,
    ) -> api::Result<()> {
        let mut state = self.phase.lock().unwrap_or_else(|p| p.into_inner());
        let previous = match *state {
            BuildState::Empty => false,
            BuildState::Ready => true,
            _ => return Err(Error::InvalidState),
        };
        *self.recording.lock().unwrap_or_else(|p| p.into_inner()) = Some(Recording {
            owner,
            values,
            #[cfg(target_vendor = "apple")]
            contributions,
        });
        *state = BuildState::Recording(owner, previous);
        Ok(())
    }
    fn abandon(&self, owner: u64) {
        let mut state = self.phase.lock().unwrap_or_else(|p| p.into_inner());
        if let BuildState::Recording(current, previous) = *state {
            if current == owner {
                let recorded = self
                    .recording
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .take();
                debug_assert!(recorded.is_some_and(|x| x.owner == owner));
                *state = if previous {
                    BuildState::Ready
                } else {
                    BuildState::Empty
                };
            }
        }
    }
    fn submit(&self, owner: u64) {
        let mut state = self.phase.lock().unwrap_or_else(|p| p.into_inner());
        if matches!(&*state, BuildState::Recording(current, _) if *current == owner) {
            *state = BuildState::Submitted;
        }
    }
    fn complete(
        &self,
        snapshot: Vec<T>,
        #[cfg(target_vendor = "apple")] contributions: Vec<u32>,
        success: bool,
    ) {
        let mut state = self.phase.lock().unwrap_or_else(|p| p.into_inner());
        if matches!(*state, BuildState::Submitted) {
            self.recording
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .take();
            if success {
                *self.values.lock().unwrap_or_else(|p| p.into_inner()) = snapshot;
                #[cfg(target_vendor = "apple")]
                {
                    *self.contributions.lock().unwrap_or_else(|p| p.into_inner()) = contributions;
                }
            }
            *state = if success {
                BuildState::Ready
            } else {
                BuildState::Failed
            };
        }
    }
}

pub(crate) struct BuildReservation {
    owner: u64,
    #[cfg(target_vendor = "apple")]
    dependencies: Arc<FrozenDependencies<MetalAcceleration>>,
    #[cfg(target_vendor = "apple")]
    snapshot: Vec<MetalAcceleration>,
    #[cfg(target_vendor = "apple")]
    contributions: Vec<u32>,
    #[cfg(target_vendor = "apple")]
    publication: super::super::submit::Completion,
}
impl BuildReservation {
    #[cfg(target_vendor = "apple")]
    pub(crate) fn new(
        owner: u64,
        dependencies: Arc<FrozenDependencies<MetalAcceleration>>,
        snapshot: Vec<MetalAcceleration>,
        contributions: Vec<u32>,
    ) -> Self {
        Self {
            owner,
            dependencies,
            snapshot,
            contributions,
            publication: super::super::submit::Completion::new(),
        }
    }
    #[cfg(target_vendor = "apple")]
    pub(crate) fn submit(
        &self,
        command: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    ) {
        use objc2_metal::{MTLCommandBuffer as _, MTLCommandBufferStatus};
        self.dependencies.submit(self.owner);
        let (dependencies, snapshot, contributions, publication) = (
            self.dependencies.clone(),
            self.snapshot.clone(),
            self.contributions.clone(),
            self.publication.clone(),
        );
        let callback = block2::RcBlock::new(
            move |completed: std::ptr::NonNull<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
            >| {
                // SAFETY: Metal invokes this callback with its live terminal command buffer.
                let success =
                    unsafe { completed.as_ref() }.status() == MTLCommandBufferStatus::Completed;
                dependencies.complete(snapshot.clone(), contributions.clone(), success);
                publication.publish(if success {
                    Ok(())
                } else {
                    Err(Error::Backend("Metal TLAS build command failed".into()))
                });
            },
        );
        // SAFETY: Metal copies this retaining callback before returning.
        unsafe { command.addCompletedHandler(block2::RcBlock::as_ptr(&callback).cast()) };
    }
    #[cfg(target_vendor = "apple")]
    pub(crate) fn publication_completion(&self) -> super::super::submit::Completion {
        self.publication.clone()
    }
    pub(crate) fn completion(&self) -> BuildCompletion {
        BuildCompletion
    }
}
impl Drop for BuildReservation {
    fn drop(&mut self) {
        #[cfg(target_vendor = "apple")]
        self.dependencies.abandon(self.owner);
    }
}

/// Compatibility completion carried by existing aggregation; native publication
/// is represented separately by `publication_completion`.
pub(crate) struct BuildCompletion;
impl BuildCompletion {
    pub(crate) fn complete(&self, _: bool) {}
}

#[cfg(all(test, not(target_vendor = "apple")))]
mod tests {
    use super::FrozenDependencies;
    use std::sync::{Arc, Barrier};

    #[test]
    fn replacement_publishes_only_after_completion() {
        let dependencies = FrozenDependencies::default();
        assert!(dependencies.freeze(1, vec![1, 2]).is_ok());
        dependencies.submit(1);
        dependencies.complete(vec![1, 2], true);
        assert!(dependencies.freeze(2, vec![3]).is_ok());
        assert_eq!(
            dependencies
                .with_snapshot(2, |values| values.to_vec())
                .unwrap(),
            [3]
        );
    }

    #[test]
    fn abandoned_recording_releases_its_snapshot() {
        let dependencies = FrozenDependencies::default();
        let value = Arc::new(());
        let weak = Arc::downgrade(&value);
        dependencies.freeze(7, vec![value]).unwrap();
        dependencies.abandon(7);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn only_one_recording_owner_wins() {
        let dependencies = Arc::new(FrozenDependencies::default());
        let barrier = Arc::new(Barrier::new(3));
        let workers: Vec<_> = (0..2)
            .map(|id| {
                let (dependencies, barrier) = (dependencies.clone(), barrier.clone());
                std::thread::spawn(move || {
                    barrier.wait();
                    dependencies.freeze(id + 1, vec![id])
                })
            })
            .collect();
        barrier.wait();
        let successes = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .filter(|result| result.is_ok())
            .count();
        assert_eq!(successes, 1);
    }
}
