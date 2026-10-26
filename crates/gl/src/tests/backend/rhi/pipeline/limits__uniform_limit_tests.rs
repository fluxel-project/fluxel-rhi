    use super::check_uniform_limits;

    #[test]
    fn counts_unique_points_and_each_stage_separately() {
        let shared = (0, 1, None);
        let array_first = (1, 0, Some(0));
        let array_second = (1, 0, Some(1));
        let stages = vec![
            vec![shared, shared, array_first],
            vec![shared, array_second],
        ];
        assert!(check_uniform_limits(&stages, 3, 4).is_ok());
        assert!(check_uniform_limits(&stages, 2, 4).is_err());
        assert!(check_uniform_limits(&stages, 3, 3).is_err());
    }
