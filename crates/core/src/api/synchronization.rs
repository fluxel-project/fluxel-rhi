//! Explicit cross-operation GPU synchronization vocabulary.
//!
//! RenderGraph supplies barriers and ownership transfers. Core preserves that
//! intent for backends and does not infer hazards or maintain global state.
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

/// Pure access vocabulary. RenderGraph collects uses and derives hazards;
/// RHI only consumes the already chosen transitions below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageMask(pub u64);
impl StageMask {
    pub const HOST: Self = Self(1 << 0);
    pub const COPY: Self = Self(1 << 1);
    pub const INDIRECT: Self = Self(1 << 2);
    pub const VERTEX_INPUT: Self = Self(1 << 3);
    pub const VERTEX: Self = Self(1 << 4);
    pub const FRAGMENT: Self = Self(1 << 5);
    pub const COLOR_OUTPUT: Self = Self(1 << 6);
    pub const COMPUTE: Self = Self(1 << 7);
    pub const TASK: Self = Self(1 << 8);
    pub const MESH: Self = Self(1 << 9);
    pub const RAY: Self = Self(1 << 10);
    pub const ACCELERATION: Self = Self(1 << 11);
    pub const ALL_COMMANDS: Self = Self(u64::MAX);
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessMask(pub u64);
impl AccessMask {
    pub const HOST_READ: Self = Self(1 << 0);
    pub const HOST_WRITE: Self = Self(1 << 1);
    pub const COPY_READ: Self = Self(1 << 2);
    pub const COPY_WRITE: Self = Self(1 << 3);
    pub const VERTEX_READ: Self = Self(1 << 4);
    pub const INDEX_READ: Self = Self(1 << 5);
    pub const UNIFORM_READ: Self = Self(1 << 6);
    pub const SHADER_READ: Self = Self(1 << 7);
    pub const SHADER_WRITE: Self = Self(1 << 8);
    pub const INDIRECT_READ: Self = Self(1 << 9);
    pub const COLOR_READ: Self = Self(1 << 10);
    pub const COLOR_WRITE: Self = Self(1 << 11);
    pub const DEPTH_READ: Self = Self(1 << 12);
    pub const DEPTH_WRITE: Self = Self(1 << 13);
    pub const STENCIL_READ: Self = Self(1 << 14);
    pub const STENCIL_WRITE: Self = Self(1 << 15);
    pub const ACCEL_READ: Self = Self(1 << 16);
    pub const ACCEL_WRITE: Self = Self(1 << 17);
    pub const QUERY_RESOLVE_WRITE: Self = Self(1 << 18);
    pub const RAY_TRACING_SHADER_DATA_READ: Self = Self(1 << 19);
    pub const ALL: Self = Self(u64::MAX);
}
macro_rules! mask_ops {
    ($t:ty) => {
        impl $t {
            pub const EMPTY: Self = Self(0);
            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }
            pub const fn contains(self, other: Self) -> bool {
                (self.0 & other.0) == other.0
            }
        }
        impl std::ops::BitOr for $t {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                self.union(rhs)
            }
        }
        impl std::ops::BitOrAssign for $t {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }
    };
}
mask_ops!(StageMask);
mask_ops!(AccessMask);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessScope {
    pub stages: StageMask,
    pub accesses: AccessMask,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferUses {
    pub stages: StageMask,
    pub accesses: AccessMask,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureUses {
    pub stages: StageMask,
    pub accesses: AccessMask,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryAccess {
    Write,
    ResolveRead,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureUseIntent {
    ShaderRead,
    ShaderReadWrite,
    ColorAttachment,
    DepthStencilRead,
    DepthStencilWrite,
    CopySrc,
    CopyDst,
    ResolveSrc,
    ResolveDst,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    /// Initial state of a newly created subresource with undefined contents.
    /// Use only before its first GPU use; this is not a request to discard an
    /// arbitrary previously used layout.
    Undefined,
    Common,
    General,
    ShaderReadOnly,
    CopySrc,
    CopyDst,
    /// Color attachment used as the source of a multisample resolve.
    /// Use COLOR_OUTPUT and COLOR_READ in the corresponding TextureUses.
    ResolveSrc,
    /// Color attachment used as the destination of a multisample resolve.
    /// Use COLOR_OUTPUT and COLOR_WRITE in the corresponding TextureUses.
    ResolveDst,
    ColorAttachment,
    DepthStencilAttachment,
    DepthReadOnly,
    Present,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateTransition<T> {
    pub from: T,
    pub to: T,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnershipTransfer {
    Release { from: LaneId, to: LaneId },
    Acquire { from: LaneId, to: LaneId },
}

pub struct BufferSlice<'a, B: Backend> {
    pub buffer: &'a Buffer<B>,
    pub range: Range<u64>,
}
pub struct Subresources<'a, B: Backend> {
    pub texture: &'a Texture<B>,
    pub aspects: TextureAspects,
    pub mips: Range<u32>,
    pub layers: Range<u32>,
}
pub struct BufferBarrier<'a, B: Backend> {
    pub target: BufferSlice<'a, B>,
    pub usage: StateTransition<BufferUses>,
    pub ownership: Option<OwnershipTransfer>,
}
pub struct TextureBarrier<'a, B: Backend> {
    pub target: Subresources<'a, B>,
    pub usage: StateTransition<TextureUses>,
    pub layout: Option<(ImageLayout, ImageLayout)>,
    pub ownership: Option<OwnershipTransfer>,
}
pub enum AliasedResource<'a, B: Backend> {
    Buffer(&'a Buffer<B>),
    Texture(&'a Texture<B>),
    Acceleration(&'a AccelerationStructure<B>),
}
pub enum ResourceTransition<'a, B: Backend> {
    Global {
        before: AccessScope,
        after: AccessScope,
    },
    Buffer(BufferBarrier<'a, B>),
    Texture(TextureBarrier<'a, B>),
    Acceleration {
        target: &'a AccelerationStructure<B>,
        before: AccessScope,
        after: AccessScope,
    },
    Aliasing {
        before: Option<AliasedResource<'a, B>>,
        after: Option<AliasedResource<'a, B>>,
    },
}
impl<B: Backend> ResourceTransition<'_, B> {
    pub(crate) fn valid_for(&self, device: DeviceId, lane: LaneId) -> bool {
        let ownership_ok = |o: Option<OwnershipTransfer>| {
            // Ownership is directional: release is issued by the source lane
            // and acquire by the destination. Core checks endpoints but leaves
            // ordering and memory scopes to the compiled plan.
            o.is_none_or(|x| match x {
                OwnershipTransfer::Release { from, to } => from == lane && to.device == device,
                OwnershipTransfer::Acquire { from, to } => to == lane && from.device == device,
            })
        };
        let resource_ok = |r: &AliasedResource<'_, B>| match r {
            AliasedResource::Buffer(x) => x.device == device,
            AliasedResource::Texture(x) => x.device == device,
            AliasedResource::Acceleration(x) => x.device == device,
        };
        match self {
            Self::Global { .. } => true,
            Self::Buffer(x) => x.target.buffer.device == device && ownership_ok(x.ownership),
            Self::Texture(x) => x.target.texture.device == device && ownership_ok(x.ownership),
            Self::Acceleration { target, .. } => target.device == device,
            Self::Aliasing { before, after } => {
                before.as_ref().is_none_or(&resource_ok) && after.as_ref().is_none_or(resource_ok)
            }
        }
    }
}
