//! DXGI adapter enumeration ordering and requested-adapter rejection tests.

use super::*;
use std::{
    future::Future,
    task::{Context, Poll, Waker},
};

#[test]
fn enumerate_exposes_every_dxgi_ordinal_in_order() -> Result<()> {
    let instance = Dx12Instance::new()?.instance()?;
    let adapters = instance.enumerate()?;
    for (ordinal, adapter) in adapters.iter().enumerate() {
        assert_eq!(adapter.id.ordinal, ordinal as u64);
    }
    Ok(())
}

#[test]
fn open_rejects_an_identifier_for_an_unenumerated_ordinal() -> Result<()> {
    let instance = Dx12Instance::new()?.instance()?;
    let adapters = instance.enumerate()?;
    let Some(last) = adapters.last() else {
        // A system with no DXGI adapters has no adapter identity to mutate.
        return Ok(());
    };
    let requested = AdapterId {
        instance: last.id.instance,
        ordinal: last.id.ordinal + 1,
    };
    let mut open = std::pin::pin!(instance.open(DeviceRequest {
        adapter: requested,
        required_features: Vec::new(),
    }));
    let mut context = Context::from_waker(Waker::noop());
    assert!(matches!(
        open.as_mut().poll(&mut context),
        Poll::Ready(Err(Error::Unsupported(REQUESTED_ADAPTER_UNAVAILABLE)))
    ));
    Ok(())
}
