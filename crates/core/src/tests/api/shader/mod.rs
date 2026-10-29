//! Shared shader-route fixture and behavior-grouped Core API tests.

use super::super::tests::ray_validation::Mock;
use super::*;
use std::{
    cell::Cell,
    future::{Ready, ready},
    task::{Context, Poll, Waker},
};
thread_local! { static ROUTES: Cell<usize> = const { Cell::new(0) }; }
impl ShaderBackend for Mock {
    type DirectFuture<'a> = Ready<Result<()>>;
    type NagaFuture<'a> = Ready<Result<()>>;
    fn shader_route(_: &(), _: &ShaderDesc) -> ShaderRoute {
        ROUTES.with(|v| v.set(v.get() + 1));
        ShaderRoute::Direct(ShaderTarget {
            format: ShaderFormat::Dxil,
            environment: String::new(),
            abi_key: String::new(),
        })
    }
    fn compile_direct<'a>(
        _: &'a (),
        _: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::DirectFuture<'a> {
        ready(Ok(()))
    }
    fn compile_naga<'a>(_: &'a (), _: &'a ShaderDesc, _: &'a ShaderTarget) -> Self::NagaFuture<'a> {
        unreachable!()
    }
}

#[path = "float32_atomics.rs"]
mod float32_atomics;
#[path = "image_atomics.rs"]
mod image_atomics;
#[path = "int64_buffer.rs"]
mod int64_buffer;
#[path = "int64_workgroup.rs"]
mod int64_workgroup;
#[path = "memory_matrix.rs"]
mod memory_matrix;
#[path = "stage_features.rs"]
mod stage_features;
