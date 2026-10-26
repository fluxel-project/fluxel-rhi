    use super::*;

    #[test]
    fn evidence_starts_without_completion_claim() {
        let evidence = GlWorkloadEvidence::new(
            GlContextEvidence {
                profile: "test".into(),
                renderer: "test".into(),
                extensions: Vec::new(),
                extent: [1, 1],
            },
            "baseline",
        );
        assert!(!evidence.completion_observed);
        assert!(evidence.rgba8.is_none());
    }
