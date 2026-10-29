//! Shared state and surface lease witnesses for Core contract tests.
use super::*;

thread_local! {
    pub(crate) static ABANDONED_STEPS: Cell<usize> = const { Cell::new(0) };
    pub(crate) static DRAW_CALLS: Cell<usize> = const { Cell::new(0) };
    pub(crate) static CLEAR_RECT_CALLS: Cell<usize> = const { Cell::new(0) };
    pub(crate) static DEPTH_BIAS_VALUES: RefCell<Vec<DynamicDepthBias>> = const { RefCell::new(Vec::new()) };
    pub(crate) static DEBUG_PUSHES: Cell<usize> = const { Cell::new(0) };
    pub(crate) static DEBUG_POPS: Cell<usize> = const { Cell::new(0) };
    pub(crate) static GL_TRANSITIONS: Cell<usize> = const { Cell::new(0) };
    pub(crate) static GL_CHECKPOINTS: Cell<usize> = const { Cell::new(0) };
    pub(crate) static GL_EMPTY_STEP_CHECKPOINTS: Cell<bool> = const { Cell::new(false) };
    pub(crate) static GL_CLOSE_FAILURE: Cell<u8> = const { Cell::new(0) };
    pub(crate) static GL_WAIT_REJECT: Cell<bool> = const { Cell::new(false) };
    pub(crate) static GL_CHECKPOINT_UNKNOWN: Cell<bool> = const { Cell::new(false) };
    pub(crate) static GL_EMERGENCY_UNKNOWN: Cell<bool> = const { Cell::new(false) };
    pub(crate) static GL_DEVICE: Cell<Option<DeviceId>> = const { Cell::new(None) };
    pub(crate) static GL_ABANDONED: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: the unit test lease has no native state and always accepts handoff.
unsafe impl NativeImageLease for () {
    fn try_handoff(&mut self, _: u64) -> Result<()> {
        Ok(())
    }
    fn is_current(&self, _: u64) -> bool {
        true
    }
}
pub(crate) struct TestLease {
    pub(crate) current: Arc<AtomicU64>,
    pub(crate) handed_off: bool,
}
// SAFETY: atomic generation and `handed_off` make the test lease single-use.
unsafe impl NativeImageLease for TestLease {
    fn try_handoff(&mut self, generation: u64) -> Result<()> {
        if !self.is_current(generation) {
            return Err(Error::SurfaceOutdated);
        }
        self.handed_off = true;
        Ok(())
    }
    fn is_current(&self, generation: u64) -> bool {
        self.current.load(Ordering::Acquire) == generation && !self.handed_off
    }
}
