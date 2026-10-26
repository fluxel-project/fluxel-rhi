    use super::GlFormat as F;
    use crate::backend::api::GlTextureAspect as A;

    #[test]
    fn blits_obey_color_numeric_class_and_exact_depth_stencil_format() {
        assert!(F::Rgba8Unorm.blit_compatible(F::R16Float, A::Color));
        assert!(F::R8Uint.blit_compatible(F::Rgba16Uint, A::Color));
        assert!(!F::R8Uint.blit_compatible(F::Rgba16Sint, A::Color));
        assert!(!F::R8Uint.blit_compatible(F::Rgba8Unorm, A::Color));
        assert!(F::Depth24PlusStencil8.blit_compatible(F::Depth24PlusStencil8, A::StencilOnly));
        assert!(!F::Depth16Unorm.blit_compatible(F::Depth24Unorm, A::DepthOnly));
        assert!(!F::Depth16Unorm.blit_compatible(F::Depth16Unorm, A::StencilOnly));
    }
