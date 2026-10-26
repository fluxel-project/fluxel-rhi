//! Typed invalidation events.  Retirement is dispatched before native deletion.
use super::{DirtyDomains, StateDomain};
use crate::backend::api::{
    BufferId, FramebufferId, ProgramId, QueryId, RenderbufferId, SamplerId, TextureId,
    VertexArrayId,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StateEvent {
    BufferRetired(BufferId),
    TextureRetired(TextureId),
    RenderbufferRetired(RenderbufferId),
    FramebufferRetired(FramebufferId),
    QueryRetired(QueryId),
    SamplerRetired(SamplerId),
    ProgramRetired(ProgramId),
    VertexArrayRetired(VertexArrayId),
    /// A partial native failure makes the named domain unknown.
    DomainFailed(StateDomain),
    ScopedRawAccess(ScopedRawAccess),
    /// A lost context is terminal. Restoration creates a new device, resource
    /// allocation table, and state mirror instead of making old identities
    /// valid again.
    ContextLost,
}
impl StateEvent {
    pub(crate) const fn domains(self) -> DirtyDomains {
        match self {
            Self::BufferRetired(_) => DirtyDomains::of(StateDomain::Geometry),
            Self::TextureRetired(_) | Self::SamplerRetired(_) => {
                DirtyDomains::of(StateDomain::Bindings)
            }
            Self::RenderbufferRetired(_) | Self::FramebufferRetired(_) => {
                DirtyDomains::of(StateDomain::PassFramebuffer)
            }
            Self::QueryRetired(_) => DirtyDomains::of(StateDomain::Query),
            Self::ProgramRetired(_) => DirtyDomains::of(StateDomain::RasterPipeline),
            Self::VertexArrayRetired(_) => DirtyDomains::of(StateDomain::Geometry),
            Self::DomainFailed(d) => DirtyDomains::of(d),
            Self::ScopedRawAccess(access) => access.domains(),
            Self::ContextLost => DirtyDomains::ALL,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ScopedRawAccess(Option<DirtyDomains>);
impl ScopedRawAccess {
    pub(crate) const fn declaring(domains: DirtyDomains) -> Self {
        Self(Some(domains))
    }
    pub(crate) const fn all() -> Self {
        Self(None)
    }
    pub(crate) const fn domains(self) -> DirtyDomains {
        match self.0 {
            Some(value) => value,
            None => DirtyDomains::ALL,
        }
    }
}
#[cfg(test)]
#[path = "../../tests/backend/state/event__tests.rs"]
mod tests;
