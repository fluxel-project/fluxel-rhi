use super::*;

#[cfg(all(test, feature = "native-gles-egl"))]
#[test]
#[ignore = "requires a GLES 3.1 hardware context"]
fn gles_failure_injection_restores_bindings_and_deletes_scratch() {
    use crate::backend::{
        api::{ContextEpoch, ContextStamp, DeviceIdentity},
        native::egl::{EglGlesContext, EglGlesVersion, EglPbufferSize},
    };

    use super::super::scratch::IndirectCountScratchPool;

    let stamp = ContextStamp::new(DeviceIdentity::new(999).unwrap(), ContextEpoch::INITIAL);
    let mut context =
        EglGlesContext::new_pbuffer(stamp, EglPbufferSize::PROBE, EglGlesVersion::V3_1).unwrap();
    context.make_current().unwrap();
    let gl = Rc::new(context.load_glow().unwrap());
    let pool = IndirectCountScratchPool::default();
    // SAFETY: this test owns the current pbuffer context and every name below.
    unsafe {
        let generic = gl.create_buffer().unwrap();
        let indexed_zero = gl.create_buffer().unwrap();
        let indexed_one = gl.create_buffer().unwrap();
        let copy_read = gl.create_buffer().unwrap();
        let copy_write = gl.create_buffer().unwrap();
        let indirect = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(generic));
        gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(indexed_zero));
        gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 1, Some(indexed_one));
        // bind_buffer_base also updates GL's generic SSBO binding. Rebind the
        // distinct generic sentinel after both indexed slots so the snapshot
        // below has the state this fixture expects to restore.
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(generic));
        gl.bind_buffer(glow::COPY_READ_BUFFER, Some(copy_read));
        gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(copy_write));
        gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(indirect));

        let binding = |target| gl.get_parameter_i32(target) as u32;
        assert_eq!(
            binding(glow::SHADER_STORAGE_BUFFER_BINDING),
            generic.0.get(),
            "fixture baseline generic SSBO"
        );
        assert_eq!(
            gl.get_parameter_indexed_i32(glow::SHADER_STORAGE_BUFFER_BINDING, 0) as u32,
            indexed_zero.0.get(),
            "fixture baseline indexed SSBO 0"
        );
        assert_eq!(
            gl.get_parameter_indexed_i32(glow::SHADER_STORAGE_BUFFER_BINDING, 1) as u32,
            indexed_one.0.get(),
            "fixture baseline indexed SSBO 1"
        );

        let (scratch, pooled) = {
            let mut transaction =
                IndirectCountTransaction::capture(gl.clone(), None, "failure-injection").unwrap();
            let lease = pool
                .lease(gl.clone(), [64, 60, 4], "failure-injection")
                .unwrap();
            let pooled = lease.buffers();
            transaction.retain_scratch(lease, 128);
            for buffer in pooled {
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
            }
            let scratch = gl.create_buffer().unwrap();
            transaction.retain_temporary(scratch);
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(scratch));
            gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(scratch));
            gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 1, Some(scratch));
            gl.bind_buffer(glow::COPY_READ_BUFFER, Some(scratch));
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(scratch));
            gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(scratch));
            let injected: Result<(), ()> = Err(());
            assert!(injected.is_err());
            (scratch, pooled)
        };

        assert_eq!(
            binding(glow::SHADER_STORAGE_BUFFER_BINDING),
            generic.0.get()
        );
        assert_eq!(binding(glow::COPY_READ_BUFFER_BINDING), copy_read.0.get());
        assert_eq!(binding(glow::COPY_WRITE_BUFFER_BINDING), copy_write.0.get());
        assert_eq!(
            binding(glow::DRAW_INDIRECT_BUFFER_BINDING),
            indirect.0.get()
        );
        assert_eq!(
            gl.get_parameter_indexed_i32(glow::SHADER_STORAGE_BUFFER_BINDING, 0) as u32,
            indexed_zero.0.get()
        );
        assert_eq!(
            gl.get_parameter_indexed_i32(glow::SHADER_STORAGE_BUFFER_BINDING, 1) as u32,
            indexed_one.0.get()
        );
        assert!(!gl.is_buffer(scratch));
        assert!(gl.is_buffer(pooled[0]), "rollback returns the pooled set");
        let reuse = pool
            .lease(gl.clone(), [64, 60, 4], "failure-injection")
            .unwrap();
        assert_eq!(
            reuse.buffers(),
            pooled,
            "next lowering reuses rollback scratch"
        );
        reuse.release(128);
        pool.clear(&gl);
        assert!(
            !gl.is_buffer(pooled[0]),
            "current-context teardown deletes returned scratch"
        );
        assert_eq!(gl.get_error(), glow::NO_ERROR);
        for buffer in [
            generic,
            indexed_zero,
            indexed_one,
            copy_read,
            copy_write,
            indirect,
        ] {
            gl.delete_buffer(buffer);
        }
    }
}
