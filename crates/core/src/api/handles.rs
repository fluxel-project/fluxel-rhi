//! Typed public wrappers around backend-native resource handles.
//!
//! Wrappers preserve device provenance and keep native types out of renderer
//! code. Native validity and destruction remain backend responsibilities.
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
pub struct Texture<B: Backend> {
    pub(crate) device: DeviceId,
    pub(crate) native: B::NativeTexture,
    pub(crate) desc: Option<TextureDesc>,
    pub(crate) native_transient_usage: bool,
}
impl<B: Backend> Clone for Texture<B> {
    fn clone(&self) -> Self {
        Self {
            device: self.device,
            native: self.native.clone(),
            desc: self.desc.clone(),
            native_transient_usage: self.native_transient_usage,
        }
    }
}
impl<B: Backend> Texture<B> {
    /// Wraps a native texture whose full RHI descriptor is unavailable.
    ///
    /// # Safety
    /// The native handle must belong to this device and object kind, and its
    /// producer must ensure it was not created with transient attachment
    /// intent. This constructor leaves descriptor provenance unknown and thus
    /// cannot enforce transient-only use restrictions.
    pub unsafe fn from_native(device: DeviceId, native: B::NativeTexture) -> Self {
        Self {
            device,
            native,
            desc: None,
            native_transient_usage: false,
        }
    }
    pub fn device(&self) -> DeviceId {
        self.device
    }
    #[doc(hidden)]
    pub unsafe fn native_ref(&self) -> &B::NativeTexture {
        &self.native
    }
    /// Whether this exact transient attachment was created through the
    /// backend's native transient mode (such as lazy allocation or a native
    /// transient usage flag). This does not prove physical tile-memory use.
    pub fn uses_native_transient_attachment(&self) -> bool {
        self.native_transient_usage
    }
}
pub struct TextureView<B: Backend> {
    pub(crate) device: DeviceId,
    pub(crate) native: B::NativeTextureView,
    pub(crate) texture_desc: Option<TextureDesc>,
}
impl<B: Backend> Clone for TextureView<B> {
    fn clone(&self) -> Self {
        Self {
            device: self.device,
            native: self.native.clone(),
            texture_desc: self.texture_desc.clone(),
        }
    }
}
impl<B: Backend> TextureView<B> {
    /// Wraps a native view whose source texture descriptor is unavailable.
    ///
    /// # Safety
    /// The native handle must belong to this device and object kind. The
    /// producer must ensure its source is not a transient attachment; this
    /// constructor cannot establish that provenance for Core validation.
    pub unsafe fn from_native(device: DeviceId, native: B::NativeTextureView) -> Self {
        Self {
            device,
            native,
            texture_desc: None,
        }
    }
    pub fn device(&self) -> DeviceId {
        self.device
    }
    #[doc(hidden)]
    pub unsafe fn native_ref(&self) -> &B::NativeTextureView {
        &self.native
    }
}
handle!(Buffer, NativeBuffer);
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
    /// # Safety
    /// Use only in the owning backend on a permitted thread; the returned
    /// reference must not outlive this layout or mutate native state directly.
    #[doc(hidden)]
    pub unsafe fn native_ref(&self) -> &B::NativeBindGroupLayout {
        &self.native
    }
}
handle!(BindGroup, NativeBindGroup);
handle!(PipelineLayout, NativePipelineLayout);
handle!(Shader, NativeShader);
pub struct RasterPipeline<B: Backend> {
    pub(crate) device: DeviceId,
    pub(crate) native: B::NativeRasterPipeline,
    pub(crate) dynamic_depth_bias: Option<DynamicDepthBias>,
}
impl<B: Backend> Clone for RasterPipeline<B> {
    fn clone(&self) -> Self {
        Self {
            device: self.device,
            native: self.native.clone(),
            dynamic_depth_bias: self.dynamic_depth_bias,
        }
    }
}
impl<B: Backend> RasterPipeline<B> {
    /// # Safety
    /// The native handle must belong to this device and object kind.
    pub unsafe fn from_native(device: DeviceId, native: B::NativeRasterPipeline) -> Self {
        Self {
            device,
            native,
            dynamic_depth_bias: None,
        }
    }
    pub(crate) unsafe fn from_native_with_dynamic_depth_bias(
        device: DeviceId,
        native: B::NativeRasterPipeline,
        dynamic_depth_bias: Option<DynamicDepthBias>,
    ) -> Self {
        Self {
            device,
            native,
            dynamic_depth_bias,
        }
    }
    pub fn device(&self) -> DeviceId {
        self.device
    }
    /// Native lowering bridge only. The caller must not mutate native GPU
    /// state outside its backend command/ownership contract.
    /// # Safety: use only on the owning Device and permitted thread.
    #[doc(hidden)]
    pub unsafe fn native_ref(&self) -> &B::NativeRasterPipeline {
        &self.native
    }
}
handle!(ComputePipeline, NativeComputePipeline);
handle!(MeshPipeline, NativeMeshPipeline);
handle!(RayPipeline, NativeRayPipeline);
handle!(PipelineCache, NativePipelineCache);
handle!(AccelerationStructure, NativeAcceleration);
handle!(SparseAllocation, NativeSparseAllocation);
