
use super::*;
use crate::backend::{features, profile::GlProfile};
use std::collections::BTreeSet;

fn compressed_entries() -> BTreeSet<Fn> {
    BTreeSet::from([Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d])
}

#[test]
fn etc2_is_gles30_and_gl43_core_but_not_webgl2_by_inference() {
    let functions = compressed_entries();
    let none = BTreeSet::new();
    assert!(features::supports(
        F::CompressionEtc2Eac,
        GlProfile::Embedded { major: 3, minor: 0 },
        &none,
        &functions
    ));
    assert!(features::supports(
        F::CompressionEtc2Eac,
        GlProfile::Desktop { major: 4, minor: 3 },
        &none,
        &functions
    ));
    assert!(!features::supports(
        F::CompressionEtc2Eac,
        GlProfile::WebGl2,
        &none,
        &functions
    ));
}

#[test]
fn bc_aggregate_requires_every_bc_family_and_both_upload_tokens() {
    // RGTC is desktop core since GL 3.0 and BPTC since GL 4.2.  Test the
    // extension fallback below 4.2 rather than pretending those core
    // families still need their extension tokens on a GL 4.2 context.
    let profile = GlProfile::Desktop { major: 4, minor: 1 };
    let mut extensions = BTreeSet::from([E::ExtTextureCompressionS3tc]);
    assert!(!features::supports(
        F::CompressionBc,
        profile,
        &extensions,
        &compressed_entries()
    ));
    extensions.insert(E::ArbTextureCompressionBptc);
    assert!(features::supports(
        F::CompressionBc,
        profile,
        &extensions,
        &compressed_entries()
    ));
    assert!(!features::supports(
        F::CompressionBc,
        profile,
        &extensions,
        &BTreeSet::from([Fn::CompressedTexImage2d])
    ));

    // On GL 4.2, S3TC is the only BC family that still needs an
    // extension route: RGTC and BPTC are both core.
    assert!(features::supports(
        F::CompressionBc,
        GlProfile::Desktop { major: 4, minor: 2 },
        &BTreeSet::from([E::ExtTextureCompressionS3tc]),
        &compressed_entries()
    ));
}
