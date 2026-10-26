    use super::{
        GlBindingLimits, GlBindingValidationError, validate_texture_unit,
        validate_uniform_buffer_binding, validate_uniform_range,
    };
    use crate::backend::api::{BufferId, ContextEpoch, ContextStamp, DeviceIdentity};

    fn limits() -> GlBindingLimits {
        GlBindingLimits {
            max_texture_units: 16,
            max_uniform_buffer_bindings: 24,
            uniform_buffer_offset_alignment: 256,
        }
    }

    fn buffer() -> BufferId {
        BufferId::new(
            ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL),
            0,
            0,
        )
    }

    #[test]
    fn texture_units_are_bounded_by_the_discovered_count() {
        assert_eq!(validate_texture_unit(15, limits()), Ok(()));
        assert_eq!(
            validate_texture_unit(16, limits()),
            Err(GlBindingValidationError::TextureUnitOutOfBounds)
        );
    }

    #[test]
    fn uniform_slots_and_offsets_are_checked_before_ranges() {
        let bound = buffer();
        assert_eq!(
            validate_uniform_buffer_binding(24, Some(bound), 0, 16, limits()),
            Err(GlBindingValidationError::UniformIndexOutOfBounds)
        );
        assert_eq!(
            validate_uniform_buffer_binding(0, Some(bound), 3, 16, limits()),
            Err(GlBindingValidationError::UniformOffsetMisaligned)
        );
        assert_eq!(
            validate_uniform_buffer_binding(0, None, 0, 256, limits()),
            Err(GlBindingValidationError::UniformUnbindCarriesRange)
        );
        assert_eq!(
            validate_uniform_buffer_binding(0, None, 0, 0, limits()),
            Ok(())
        );
    }

    #[test]
    fn uniform_ranges_resolve_size_zero_to_the_allocation_end() {
        assert_eq!(validate_uniform_range(0, 0, 512), Ok(()));
        assert_eq!(validate_uniform_range(256, 256, 512), Ok(()));
        assert_eq!(
            validate_uniform_range(512, 0, 512),
            Err(GlBindingValidationError::UniformOffsetExceedsAllocation)
        );
        assert_eq!(
            validate_uniform_range(0, 513, 512),
            Err(GlBindingValidationError::UniformRangeOutOfBounds)
        );
    }

    #[test]
    fn uniform_ranges_keep_a_64_bit_native_address() {
        let offset = 4_u64 * 1024 * 1024 * 1024 + 256;
        assert_eq!(
            validate_uniform_buffer_binding(0, Some(buffer()), offset, 256, limits()),
            Ok(())
        );
        assert_eq!(validate_uniform_range(offset, 256, offset + 512), Ok(()));
    }
