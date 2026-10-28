use super::{
    GlDispatchIndirectAbi, GlDispatchIndirectCommand, GlIndirectAbi, GlIndirectCommandRange,
};
use crate::backend::api::{
    BufferId, ContextEpoch, ContextStamp, DeviceIdentity, GlBufferRange, GlError,
};

fn stamp() -> ContextStamp {
    ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL)
}
fn range(offset: u64, size: u64) -> GlBufferRange {
    GlBufferRange {
        buffer: BufferId::new(stamp(), 1, 1),
        offset,
        size,
    }
}

#[test]
fn rejects_misaligned_indirect_offset() {
    let s = stamp();
    let c = GlIndirectCommandRange {
        range: GlBufferRange {
            buffer: BufferId::new(s, 1, 1),
            offset: 0,
            size: 16,
        },
        command_offset: 2,
        draw_count: 1,
        stride: 0,
        abi: GlIndirectAbi::NonIndexed,
    };
    assert!(c.validate("draw_indirect").is_err());
}

/// The dispatch record is the WebGPU dispatch argument layout: three u32
/// work-group counts, twelve bytes, no padding word.
#[test]
fn dispatch_record_is_three_work_group_counts() {
    assert_eq!(GlDispatchIndirectAbi::SIZE, 12);
    assert_eq!(std::mem::align_of::<GlDispatchIndirectAbi>(), 4);
    let record = GlDispatchIndirectAbi {
        workgroup_count: [2, 3, 4],
    };
    assert_eq!(record.workgroup_count, [2, 3, 4]);
}

#[test]
fn accepts_a_whole_dispatch_record_inside_its_range() {
    let whole = GlDispatchIndirectCommand {
        range: range(0, 12),
        command_offset: 0,
    };
    assert_eq!(whole.validate("dispatch-indirect"), Ok(()));
    let trailing = GlDispatchIndirectCommand {
        range: range(16, 24),
        command_offset: 8,
    };
    assert_eq!(trailing.validate("dispatch-indirect"), Ok(()));
}

#[test]
fn rejects_dispatch_records_that_do_not_fit_their_range() {
    let missing_word = GlDispatchIndirectCommand {
        range: range(0, 8),
        command_offset: 0,
    };
    assert!(matches!(
        missing_word.validate("dispatch-indirect"),
        Err(GlError::Validation { .. })
    ));
    let past_the_end = GlDispatchIndirectCommand {
        range: range(0, 24),
        command_offset: 16,
    };
    assert!(past_the_end.validate("dispatch-indirect").is_err());
    let misaligned = GlDispatchIndirectCommand {
        range: range(0, 24),
        command_offset: 6,
    };
    assert!(misaligned.validate("dispatch-indirect").is_err());
}

#[test]
fn rejects_dispatch_commands_outside_the_buffer_role() {
    let misaligned_range = GlDispatchIndirectCommand {
        range: range(2, 12),
        command_offset: 0,
    };
    assert!(misaligned_range.validate("dispatch-indirect").is_err());
    let empty = GlDispatchIndirectCommand {
        range: range(0, 0),
        command_offset: 0,
    };
    assert!(empty.validate("dispatch-indirect").is_err());
}

/// A record reaching past `u32`/`i32` arithmetic must fail closed instead of
/// wrapping into a plausible in-range offset.
#[test]
fn overflowing_dispatch_offsets_are_rejected_not_wrapped() {
    let overflowing = GlDispatchIndirectCommand {
        range: range(0, u64::MAX - 3),
        command_offset: u64::MAX - 3,
    };
    assert!(overflowing.validate("dispatch-indirect").is_err());
}
