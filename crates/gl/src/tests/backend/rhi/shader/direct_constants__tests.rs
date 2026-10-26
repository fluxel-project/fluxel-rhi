    use super::*;
    use crate::backend::api::{GlShaderDialect, GlShaderStage};

    fn source(text: &str) -> GlShaderSource {
        GlShaderSource {
            stage: GlShaderStage::Fragment,
            dialect: GlShaderDialect::Desktop { version: 460 },
            entry_point: "main".into(),
            source_hash: GlShaderKey([0; 32]),
            text: text.into(),
            spirv: None,
            debug_name: None,
        }
    }

    #[test]
    fn constants_are_inserted_after_version_with_stable_literals() {
        let mut source = source(
            "#version 460 core\n#extension GL_ARB_gpu_shader_fp64 : require\nvoid main() {}\n",
        );
        apply(
            &mut source,
            &[
                ("COUNT".into(), SpecializationValue::U32(4)),
                ("scale".into(), SpecializationValue::F32(1.0)),
                ("toggle".into(), SpecializationValue::Bool(true)),
            ],
        )
        .unwrap();
        assert!(source.text.starts_with(
            "#version 460 core\n#define COUNT 4u\n#define scale 1.0\n#define toggle true\n"
        ));
        assert!(source.text.contains("#extension GL_ARB_gpu_shader_fp64"));
    }

    #[test]
    fn constants_reject_injection_reserved_and_unrepresentable_values() {
        let mut source = source("#version 460 core\nvoid main() {}\n");
        assert!(
            apply(
                &mut source,
                &[("bad\n#define injected".into(), SpecializationValue::I32(1))]
            )
            .is_err()
        );
        assert!(
            apply(
                &mut source,
                &[("gl_Position".into(), SpecializationValue::I32(1))]
            )
            .is_err()
        );
        assert!(apply(&mut source, &[("wide".into(), SpecializationValue::U64(1))]).is_err());
        assert!(
            apply(
                &mut source,
                &[("nan".into(), SpecializationValue::F32(f32::NAN))]
            )
            .is_err()
        );
    }

    #[test]
    fn version_scanner_skips_license_comments_and_fake_directives() {
        let mut source =
            source("/* license\n#version 100\n*/ // header\n#version 460 core\nvoid main() {}\n");
        apply(
            &mut source,
            &[("GAIN".into(), SpecializationValue::F32(0.5))],
        )
        .unwrap();
        assert!(
            source
                .text
                .contains("#version 460 core\n#define GAIN 0.5\n")
        );
        assert!(version_directive_end("/* unclosed #version 460\n").is_none());
    }

    #[test]
    fn unterminated_version_line_is_separated_from_definitions() {
        let mut source = source("#version 460 core");
        apply(
            &mut source,
            &[("GAIN".into(), SpecializationValue::F32(0.5))],
        )
        .unwrap();
        assert_eq!(source.text, "#version 460 core\n#define GAIN 0.5\n");
    }
