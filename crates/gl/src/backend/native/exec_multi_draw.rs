//! Native multi-draw-indirect batching with GPU-only single-command fallback.
use super::{exec_compute::draw_indirect_offset, provider::NativeGlProvider};
use crate::backend::api::{
    GlCapability, GlDrawIndirectApi as _, GlError, GlFamilyApi as _, GlIndirectAbi,
    GlIndirectCommandRange,
};

impl crate::backend::api::GlMultiDrawIndirectApi for NativeGlProvider {
    fn multi_draw_indirect(&mut self, command: GlIndirectCommandRange) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "multi-draw-indirect";
        self.assert_ready(OP)?;
        command.validate(OP)?;
        let record_stride = if command.stride == 0 {
            match command.abi {
                GlIndirectAbi::NonIndexed => 16,
                GlIndirectAbi::Indexed => 20,
            }
        } else {
            command.stride
        };
        // A missing batch entry is deliberately only a native-call fallback:
        // each draw still consumes its record on the GPU and no argument bytes
        // are mapped or read by the CPU.
        let functions = self.multi_draw_indirect.filter(|_| {
            self.discovery
                .capabilities()
                .supports(GlCapability::MultiDrawIndirect)
        });
        let Some(functions) = functions else {
            for draw in 0..command.draw_count {
                self.draw_indirect(GlIndirectCommandRange {
                    command_offset: command.command_offset
                        + u64::from(draw) * u64::from(record_stride),
                    draw_count: 1,
                    ..command
                })?;
            }
            return Ok(());
        };
        self.validate_indirect_vertex_state(OP)?;
        let (program, topology) = {
            let raster = self
                .raster
                .as_ref()
                .ok_or_else(|| Self::validation(OP, "no raster pipeline is installed"))?;
            (raster.program, raster.topology)
        };
        self.validate_indirect_driver(OP, program)?;
        self.set_first_instance_uniform(program, 0)?;
        self.ensure_program(OP, program)?;
        let bound = self
            .bound_vertex_array
            .ok_or_else(|| Self::validation(OP, "no vertex array is bound"))?;
        let index = self.vertex_array(OP, bound)?.index;
        if command.abi == GlIndirectAbi::Indexed && index.is_none() {
            return Err(Self::validation(
                OP,
                "indexed indirect draw requires a bound index buffer",
            ));
        }
        // Rebasing an indexed command with a nonzero element-buffer offset is
        // a per-record GPU transform. Preserve that established route rather
        // than inventing a CPU decode or claiming it can batch untouched ABI.
        if index.is_some_and(|binding| command.abi == GlIndirectAbi::Indexed && binding.offset != 0)
        {
            for draw in 0..command.draw_count {
                self.draw_indirect(GlIndirectCommandRange {
                    command_offset: command.command_offset
                        + u64::from(draw) * u64::from(record_stride),
                    draw_count: 1,
                    ..command
                })?;
            }
            return Ok(());
        }
        let source = self.indirect_command_buffer(OP, command.range)?;
        let offset = draw_indirect_offset(command.range, command.command_offset)?;
        let mode = super::exec_raster::topology_mode(topology);
        let draw_count = i32::try_from(command.draw_count)
            .map_err(|_| Self::validation(OP, "draw count exceeds GLsizei"))?;
        let stride = i32::try_from(command.stride)
            .map_err(|_| Self::validation(OP, "stride exceeds GLsizei"))?;
        // SAFETY: all state, buffer extent, pointer-width offset, and index ABI
        // were validated above; the driver reads every command from the bound
        // indirect buffer without CPU observation.
        unsafe {
            self.gl
                .bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(source));
            if let Some(index) = index.filter(|_| command.abi == GlIndirectAbi::Indexed) {
                (functions.elements)(
                    mode,
                    super::exec_vertex::indexed_draw_type(index),
                    offset as usize as *const core::ffi::c_void,
                    draw_count,
                    stride,
                );
            } else {
                (functions.arrays)(
                    mode,
                    offset as usize as *const core::ffi::c_void,
                    draw_count,
                    stride,
                );
            }
        }
        self.defer_driver_error(OP)
    }
}
