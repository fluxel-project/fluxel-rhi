//! Range and usage checks for Metal's three-`u32` indirect dispatch record.

use super::{
    INDIRECT_ARGUMENT_BYTES, validate_indirect_dispatch, validate_indirect_dispatch_support,
};
use fluxel_rhi_core::api::{BufferUsage, Error};

#[test]
fn indirect_dispatch_requires_aligned_complete_arguments_and_usage() {
    assert!(validate_indirect_dispatch(BufferUsage::INDIRECT, 24, 4).is_ok());
    assert_eq!(
        validate_indirect_dispatch(BufferUsage::INDIRECT, 24, 2),
        Err(Error::InvalidRange)
    );
    assert_eq!(
        validate_indirect_dispatch(BufferUsage::INDIRECT, 15, 4),
        Err(Error::InvalidRange)
    );
    assert_eq!(
        validate_indirect_dispatch(BufferUsage::COPY_SRC, 24, 4),
        Err(Error::InvalidRange)
    );
    assert_eq!(
        validate_indirect_dispatch(
            BufferUsage::INDIRECT,
            u64::MAX,
            u64::MAX - (INDIRECT_ARGUMENT_BYTES - 2),
        ),
        Err(Error::InvalidRange)
    );
}

#[test]
fn indirect_dispatch_support_is_required_only_for_the_indirect_verb() {
    assert_eq!(
        validate_indirect_dispatch_support(false),
        Err(Error::Unsupported(
            "Metal indirect dispatch requires Apple3 or Mac2 GPU family"
        ))
    );
    assert_eq!(validate_indirect_dispatch_support(true), Ok(()));
}
