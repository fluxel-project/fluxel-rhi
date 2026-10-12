//! Statically typed RHI contract component.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]
use super::*;
use std::{
    future::Future,
    marker::PhantomData,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

macro_rules! handle {
    ($name:ident, $native:ident) => {
        pub struct $name<B: Backend> {
            pub(crate) device: DeviceId,
            pub(crate) native: B::$native,
        }
        impl<B: Backend> Clone for $name<B> {
            fn clone(&self) -> Self {
                Self {
                    device: self.device,
                    native: self.native.clone(),
                }
            }
        }
        impl<B: Backend> $name<B> {
            /// # Safety
            /// The native handle must belong to this device and object kind.
            pub unsafe fn from_native(device: DeviceId, native: B::$native) -> Self {
                Self { device, native }
            }
            pub fn device(&self) -> DeviceId {
                self.device
            }
            /// Native lowering bridge only. The caller must not mutate native
            /// GPU state outside its backend command/ownership contract.
            /// # Safety: use only on the owning Device and permitted thread.
            #[doc(hidden)]
            pub unsafe fn native_ref(&self) -> &B::$native {
                &self.native
            }
        }
    };
}
handle!(Buffer, NativeBuffer);
handle!(Texture, NativeTexture);
handle!(TextureView, NativeTextureView);
handle!(Sampler, NativeSampler);
pub struct BindGroupLayout<B: Backend> {
    pub(crate) device: DeviceId,
    pub(crate) native: B::NativeBindGroupLayout,
    pub(crate) entries: Arc<[BindGroupLayoutEntry]>,
}
impl<B: Backend> Clone for BindGroupLayout<B> {
    fn clone(&self) -> Self {
        Self {
            device: self.device,
            native: self.native.clone(),
            entries: self.entries.clone(),
        }
    }
}
impl<B: Backend> BindGroupLayout<B> {
    /// # Safety
    /// The native handle must belong to this device and object kind.
    pub unsafe fn from_native(
        device: DeviceId,
        native: B::NativeBindGroupLayout,
        entries: Arc<[BindGroupLayoutEntry]>,
    ) -> Self {
        Self {
            device,
            native,
            entries,
        }
    }
    pub fn device(&self) -> DeviceId {
        self.device
    }
    #[doc(hidden)]
    pub unsafe fn native_ref(&self) -> &B::NativeBindGroupLayout {
        &self.native
    }
}
handle!(BindGroup, NativeBindGroup);
handle!(PipelineLayout, NativePipelineLayout);
handle!(Shader, NativeShader);
handle!(RasterPipeline, NativeRasterPipeline);
handle!(ComputePipeline, NativeComputePipeline);
handle!(MeshPipeline, NativeMeshPipeline);
handle!(RayPipeline, NativeRayPipeline);
handle!(PipelineCache, NativePipelineCache);
handle!(AccelerationStructure, NativeAcceleration);
handle!(SparseAllocation, NativeSparseAllocation);
