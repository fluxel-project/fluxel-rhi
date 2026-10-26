//! Feature-route requirements for raster drawing.

use super::{
    GL40, GL43, GL46, GLES31, GlExtension as E, GlFeature as F, GlFunction as Fn, NEVER,
    Requirement,
};

pub(super) fn requirement(feature: F) -> Option<Requirement> {
    Some(match feature {
        F::IndirectDraw => Requirement {
            desktop_core: GL40,
            gles_core: GLES31,
            extensions: &[E::ArbDrawIndirect],
            functions: &[Fn::DrawArraysIndirect, Fn::DrawElementsIndirect],
        },
        F::MultiDrawIndirect => Requirement {
            desktop_core: GL43,
            // GLES 3.2 has no core MDI command family; do not advertise the
            // desktop ARB route through an embedded version number.
            gles_core: NEVER,
            extensions: &[E::ArbMultiDrawIndirect],
            functions: &[Fn::MultiDrawArraysIndirect, Fn::MultiDrawElementsIndirect],
        },
        F::IndirectCount => Requirement {
            desktop_core: GL46,
            gles_core: NEVER,
            extensions: &[E::ArbIndirectParameters],
            functions: &[
                Fn::MultiDrawArraysIndirectCount,
                Fn::MultiDrawElementsIndirectCount,
            ],
        },
        _ => return None,
    })
}

#[cfg(test)]
#[path = "../../tests/backend/features/draw__tests.rs"]
mod tests;
