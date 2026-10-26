//! Feature-route requirements for query operations.

use super::{GL15, GLES30, GlFeature as F, GlFunction as Fn, Requirement};

pub(super) fn requirement(feature: F) -> Option<Requirement> {
    Some(match feature {
        F::OcclusionQuery => Requirement {
            desktop_core: GL15,
            gles_core: GLES30,
            extensions: &[],
            functions: &[Fn::BeginQuery, Fn::EndQuery, Fn::GetQueryObject],
        },
        _ => return None,
    })
}

#[cfg(test)]
#[path = "../../tests/backend/features/query__tests.rs"]
mod tests;
