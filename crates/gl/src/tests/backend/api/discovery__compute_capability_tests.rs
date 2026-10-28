use super::*;

#[test]
fn compute_does_not_depend_on_storage_image_probe() {
    let fact = |probe| GlCapabilityFact {
        evidence: Some(CapabilityEvidence::Core(GlFamilyProfile::Desktop {
            major: 4,
            minor: 3,
        })),
        limits_satisfied: true,
        operation_probe: probe,
    };
    let mut caps = GlCapabilitySet::default();
    caps.facts
        .insert(GlCapability::Compute, fact(GlOperationProbe::Passed));
    caps.facts
        .insert(GlCapability::StorageImage, fact(GlOperationProbe::Failed));
    assert!(caps.supports_compute());
    assert!(!caps.supports(GlCapability::StorageImage));
    caps.facts
        .insert(GlCapability::Compute, fact(GlOperationProbe::Failed));
    caps.facts
        .insert(GlCapability::StorageImage, fact(GlOperationProbe::Passed));
    assert!(!caps.supports_compute());
    assert!(caps.supports(GlCapability::StorageImage));
}
