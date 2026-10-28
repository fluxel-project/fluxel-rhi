#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires a GLES 3.1 hardware context"]
fn gles_scratch_set_is_reused_then_deleted_at_context_teardown() {
    use std::rc::Rc;

    use crate::backend::{
        api::{ContextEpoch, ContextStamp, DeviceIdentity},
        native::egl::{EglGlesContext, EglGlesVersion, EglPbufferSize},
    };
    use glow::HasContext as _;

    use super::{IndirectCountScratchPool, RETAINED_BYTES_LIMIT};

    let stamp = ContextStamp::new(DeviceIdentity::new(998).unwrap(), ContextEpoch::INITIAL);
    let mut context =
        EglGlesContext::new_pbuffer(stamp, EglPbufferSize::PROBE, EglGlesVersion::V3_1).unwrap();
    context.make_current().unwrap();
    let gl = Rc::new(context.load_glow().unwrap());
    let pool = IndirectCountScratchPool::default();

    // SAFETY: this fixture owns the current pbuffer context.
    unsafe {
        let first = pool
            .lease(gl.clone(), [64, 60, 4], "scratch-pool-test")
            .unwrap();
        let names = first.buffers();
        for buffer in names {
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
        }
        gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
        first.release(128);

        let oversized = pool
            .lease(
                gl.clone(),
                [RETAINED_BYTES_LIMIT + 1, 0, 0],
                "scratch-pool-test",
            )
            .unwrap();
        let transient = oversized.buffers();
        for buffer in transient {
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
        }
        gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
        oversized.release(RETAINED_BYTES_LIMIT + 1);

        let second = pool
            .lease(gl.clone(), [64, 60, 4], "scratch-pool-test")
            .unwrap();
        assert_eq!(
            second.buffers(),
            names,
            "second lowering must reuse the set"
        );
        second.release(128);
        pool.clear(&gl);
        for buffer in names {
            assert!(!gl.is_buffer(buffer), "teardown must delete pooled scratch");
        }
        assert_eq!(gl.get_error(), glow::NO_ERROR);
    }
}
