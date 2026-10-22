//! Descriptor array capacity validation for DX12 tables.
use fluxel_rhi_core::api::{self, Error, Result};

pub(super) fn count(count: api::BindingCount) -> Result<u32> {
    match count {
        api::BindingCount::One => Ok(1),
        api::BindingCount::Fixed(value) if value > 0 => Ok(value),
        api::BindingCount::Fixed(_) => Err(Error::InvalidRange),
        api::BindingCount::RuntimeSized { max_count } if max_count > 0 => Ok(max_count),
        api::BindingCount::RuntimeSized { .. } => Err(Error::InvalidRange),
    }
}
