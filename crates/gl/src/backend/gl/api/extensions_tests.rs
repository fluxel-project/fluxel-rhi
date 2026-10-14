//! Tests for the private extensions.rs contract.
use super::{
    CapabilityEvidence, CoreOrExtension, ExtensionProvenance, GlExtensionSet, GlFamilyProfile,
    GlKnownExtension,
};

/// Every typed name in the inventory, so a new variant cannot be added
/// without a registry spelling that maps back to it.
const INVENTORY: [GlKnownExtension; 21] = [
    GlKnownExtension::ArbComputeShader,
    GlKnownExtension::ArbShaderStorageBufferObject,
    GlKnownExtension::ArbShaderImageLoadStore,
    GlKnownExtension::ExtDisjointTimerQueryWebgl2,
    GlKnownExtension::ExtColorBufferFloat,
    GlKnownExtension::ExtFloatBlend,
    GlKnownExtension::OesTextureFloatLinear,
    GlKnownExtension::ExtTextureFilterAnisotropic,
    GlKnownExtension::WebglMultiDraw,
    GlKnownExtension::WebglMultiDrawInstancedBaseVertexBaseInstance,
    GlKnownExtension::OvrMultiview2,
    GlKnownExtension::WebglShaderPixelLocalStorage,
    GlKnownExtension::KhrParallelShaderCompile,
    GlKnownExtension::KhrRobustness,
    GlKnownExtension::KhrDebug,
    GlKnownExtension::CompressedTextureS3tc,
    GlKnownExtension::CompressedTextureS3tcSrgb,
    GlKnownExtension::CompressedTextureBptc,
    GlKnownExtension::CompressedTextureRgtc,
    GlKnownExtension::CompressedTextureAstc,
    GlKnownExtension::CompressedTextureEtc,
];

#[test]
fn every_typed_name_maps_back_from_its_own_registry_spelling() {
    for known in INVENTORY {
        assert_eq!(
            GlKnownExtension::from_raw_name(known.raw_name()),
            Some(known),
            "{known:?}"
        );
    }
    // A name that is not in the inventory stays untyped rather than being
    // guessed at by prefix.
    assert_eq!(GlKnownExtension::from_raw_name("WEBGL_unknown_thing"), None);
    assert_eq!(GlKnownExtension::from_raw_name("webgl_multi_draw"), None);
}

/// The two inventory entries kept without a command path are reachable by
/// their exact spelling, so a context that reports one is recorded rather
/// than dropped, while nothing in the ledger can make them usable.
#[test]
fn draft_names_are_recorded_in_the_ledger_without_a_typed_route() {
    let drafts = [
        (
            "WEBGL_multi_draw_instanced_base_vertex_base_instance",
            GlKnownExtension::WebglMultiDrawInstancedBaseVertexBaseInstance,
        ),
        (
            "WEBGL_shader_pixel_local_storage",
            GlKnownExtension::WebglShaderPixelLocalStorage,
        ),
    ];
    for (name, known) in drafts {
        assert_eq!(GlKnownExtension::from_raw_name(name), Some(known), "{name}");
        let mut ledger = GlExtensionSet::default();
        ledger.report_raw(name);
        assert_eq!(ledger.raw_reported_names().collect::<Vec<_>>(), [name]);
        // Reporting is the whole of it: no row resolves against these, and
        // the draft spellings are not accepted under any other casing.
        assert_eq!(
            ledger.provenance(known),
            Some(ExtensionProvenance::Reported)
        );
        assert!(!ledger.is_acquired(known));
        assert_eq!(GlKnownExtension::from_raw_name(&name.to_lowercase()), None);
    }
}

/// Only the second multiview revision is aliased onto the typed name: the
/// first carries a shader contract this series does not implement, so it
/// stays an untyped raw name that enables nothing.
#[test]
fn only_the_second_multiview_revision_reaches_the_typed_name() {
    assert_eq!(
        GlKnownExtension::from_raw_name("GL_OVR_multiview2"),
        Some(GlKnownExtension::OvrMultiview2)
    );
    assert_eq!(GlKnownExtension::from_raw_name("GL_OVR_multiview"), None);
    let mut ledger = GlExtensionSet::default();
    ledger.report_raw("GL_OVR_multiview");
    assert_eq!(ledger.provenance(GlKnownExtension::OvrMultiview2), None);
    assert_eq!(
        ledger.raw_reported_names().collect::<Vec<_>>(),
        ["GL_OVR_multiview"]
    );
}

/// A route is legal only where its family can report the name, which is
/// what keeps a desktop context from ever resolving multiview evidence.
#[test]
fn a_typed_route_is_legal_only_inside_its_own_family() {
    let webgl = GlFamilyProfile::WebGl2;
    let embedded = GlFamilyProfile::Embedded { major: 3, minor: 1 };
    let desktop = GlFamilyProfile::Desktop { major: 4, minor: 3 };
    assert!(GlKnownExtension::WebglMultiDraw.is_legal_for(webgl));
    assert!(!GlKnownExtension::WebglMultiDraw.is_legal_for(embedded));
    assert!(!GlKnownExtension::WebglMultiDraw.is_legal_for(desktop));
    assert!(GlKnownExtension::OvrMultiview2.is_legal_for(webgl));
    assert!(GlKnownExtension::OvrMultiview2.is_legal_for(embedded));
    assert!(!GlKnownExtension::OvrMultiview2.is_legal_for(desktop));
}

/// The three evidence states a route can pass through, in order: reporting
/// enables nothing, acquiring enables a route with no probe, and a probed
/// route enables one that requires it.
#[test]
fn resolve_requires_the_exact_evidence_its_route_asks_for() {
    let profile = GlFamilyProfile::WebGl2;
    let probed_route = CoreOrExtension {
        desktop_core: None,
        embedded_core: None,
        extension: Some(GlKnownExtension::OvrMultiview2),
        extension_requires_probe: true,
    };
    let direct_route = CoreOrExtension {
        extension_requires_probe: false,
        ..probed_route
    };
    let mut ledger = GlExtensionSet::default();
    ledger.report_raw("OVR_multiview2");
    assert_eq!(probed_route.resolve(profile, &ledger), None);
    assert_eq!(direct_route.resolve(profile, &ledger), None);
    assert!(ledger.acquire(GlKnownExtension::OvrMultiview2));
    assert_eq!(
        direct_route.resolve(profile, &ledger),
        Some(CapabilityEvidence::Extension(
            GlKnownExtension::OvrMultiview2
        ))
    );
    assert_eq!(probed_route.resolve(profile, &ledger), None);
    assert!(ledger.probe(GlKnownExtension::OvrMultiview2));
    assert_eq!(
        probed_route.resolve(profile, &ledger),
        Some(CapabilityEvidence::Extension(
            GlKnownExtension::OvrMultiview2
        ))
    );
    // A failed acquisition is terminal: no later report can revive it.
    let mut failed = GlExtensionSet::default();
    failed.report_raw("OVR_multiview2");
    assert!(failed.fail(GlKnownExtension::OvrMultiview2));
    assert!(!failed.acquire(GlKnownExtension::OvrMultiview2));
    assert_eq!(direct_route.resolve(profile, &failed), None);
}

#[test]
fn compressed_extension_names_preserve_native_and_web_registry_identity() {
    let cases = [
        (
            "WEBGL_compressed_texture_s3tc",
            GlKnownExtension::CompressedTextureS3tc,
        ),
        (
            "GL_EXT_texture_compression_s3tc",
            GlKnownExtension::ExtTextureCompressionS3tc,
        ),
        (
            "GL_ARB_texture_compression_bptc",
            GlKnownExtension::ArbTextureCompressionBptc,
        ),
        (
            "GL_EXT_texture_compression_rgtc",
            GlKnownExtension::ExtTextureCompressionRgtc,
        ),
        (
            "GL_KHR_texture_compression_astc_ldr",
            GlKnownExtension::KhrTextureCompressionAstcLdr,
        ),
        (
            "GL_OES_compressed_ETC2_EAC",
            GlKnownExtension::CompressedTextureEtc,
        ),
    ];
    for (raw, expected) in cases {
        assert_eq!(
            GlKnownExtension::from_raw_name(raw),
            Some(expected),
            "{raw}"
        );
    }
    let mut ledger = GlExtensionSet::default();
    ledger.report_raw("WEBGL_compressed_texture_s3tc");
    ledger.report_raw("GL_EXT_texture_compression_s3tc");
    // The native GL extension and WebGL extension use different registries.
    // Browser capability discovery translates the WebGL token to the native
    // feature-ledger token explicitly; the raw-name parser must not silently
    // conflate the two spelling domains.
    assert!(ledger.acquire(GlKnownExtension::CompressedTextureS3tc));
    assert!(ledger.acquire(GlKnownExtension::ExtTextureCompressionS3tc));
    assert!(ledger.is_acquired(GlKnownExtension::CompressedTextureS3tc));
    assert!(ledger.is_acquired(GlKnownExtension::ExtTextureCompressionS3tc));
}
