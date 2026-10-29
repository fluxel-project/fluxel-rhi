//! Reusable Core test backends and observable SPI witness state.
pub(crate) use crate::{api::*, backend::*};
pub(crate) use std::{
    cell::{Cell, RefCell},
    future::{Future, Ready, ready},
    marker::PhantomData,
    ops::Range,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

mod deferred;
mod profiles;
mod state;
mod witnesses;

pub(crate) use profiles::{GlLocal, NativeShared};
pub(crate) use state::*;
