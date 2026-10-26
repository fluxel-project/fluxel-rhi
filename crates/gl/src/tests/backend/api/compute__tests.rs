    use super::{GlComputeLimits, GlDispatchGroups};
    use crate::backend::api::GlMemoryBarrier;
    #[test]
    fn rejects_bad_preflight_values() {
        let l = GlComputeLimits {
            max_group_count: [1; 3],
            max_group_size: [1; 3],
            max_group_invocations: 1,
        };
        assert!(GlDispatchGroups([0, 1, 1]).validate(l).is_err());
        assert!(GlMemoryBarrier(0).validate_nonempty().is_err());
    }
