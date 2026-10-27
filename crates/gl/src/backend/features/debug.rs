//! Feature-route requirements for debug output.

use super::{GL43, GLES32, GlExtension as E, GlFeature as F, GlFunction as Fn, Requirement};

pub(super) fn requirement(feature: F) -> Option<Requirement> {
    Some(match feature {
        F::DebugMarkers => Requirement {
            desktop_core: GL43,
            gles_core: GLES32,
            extensions: &[E::KhrDebug],
            functions: &[
                Fn::PushDebugGroup,
                Fn::PopDebugGroup,
                Fn::DebugMessageInsert,
            ],
        },
        _ => return None,
    })
}

#[cfg(test)]
#[path = "../../tests/backend/features/debug__tests.rs"]
mod tests;
