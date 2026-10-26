    use super::*;
    #[test]
    fn raster_stages_cannot_mix_native_artifact_formats() {
        let source = |stage| GlShaderSource {
            stage,
            dialect: GlShaderDialect::Desktop { version: 460 },
            entry_point: "main".into(),
            source_hash: GlShaderKey([0; 32]),
            text: "void main() {}".into(),
            spirv: None,
            debug_name: None,
        };
        let mut fragment = source(GlShaderStage::Fragment);
        fragment.spirv = Some(GlSpirvSource {
            words: vec![],
            specialization: vec![],
        });
        let program = GlProgramDescriptor {
            kind: GlProgramKind::Raster {
                vertex: source(GlShaderStage::Vertex),
                fragment,
            },
            layout: GlPipelineLayout { bindings: vec![] },
            debug_name: None,
        };
        assert_eq!(
            program.validate(),
            Err(GlShaderValidationError::MixedShaderFormats)
        );
    }
    #[test]
    fn fixed_one_element_assignment_keeps_array_identity() {
        let location = GlBindingLocation {
            group: 0,
            binding: 0,
        };
        let layout = GlPipelineLayout {
            bindings: vec![GlLogicalBinding {
                names: vec!["binding_0".into()],
                location,
                element: None,
                kind: GlShaderResourceKind::UniformBuffer,
                array_count: 1,
                fixed_array: true,
                pair: None,
                image_unit: None,
                sampled_type: None,
                image_type: None,
            }],
        };
        let mut reflection = GlProgramReflection {
            vertex_inputs: vec![],
            fragment_outputs: vec![],
            assignments: vec![GlExecutableBindingAssignment {
                logical: location,
                element: Some(0),
                executable: GlExecutableBindingLocation::UniformBlock(0),
            }],
        };
        assert!(reflection.validate_against(&layout).is_ok());
        reflection.assignments[0].element = Some(1);
        assert!(reflection.validate_against(&layout).is_err());
    }
    #[test]
    fn rejects_duplicate_logical_bindings() {
        let location = GlBindingLocation {
            group: 0,
            binding: 0,
        };
        let binding = GlLogicalBinding {
            element: None,
            names: vec!["a".into()],
            location,
            kind: GlShaderResourceKind::UniformBuffer,
            array_count: 1,
            fixed_array: false,
            pair: None,
            image_unit: None,
            sampled_type: None,
            image_type: None,
        };
        let layout = GlPipelineLayout {
            bindings: vec![binding.clone(), binding],
        };
        assert_eq!(
            layout.validate(),
            Err(GlShaderValidationError::DuplicateLogicalBinding)
        );
    }

    fn source(stage: GlShaderStage, dialect: GlShaderDialect) -> GlShaderSource {
        GlShaderSource {
            stage,
            dialect,
            entry_point: "main".into(),
            source_hash: GlShaderKey([7; 32]),
            text: "void main() {}".into(),
            spirv: None,
            debug_name: None,
        }
    }

    #[test]
    fn accepts_compatible_lowered_dialect_for_profile() {
        let web = source(
            GlShaderStage::Vertex,
            GlShaderDialect::Embedded { version: 300 },
        );
        assert!(web.validate_for(GlFamilyProfile::WebGl2).is_ok());
        assert_eq!(
            web.validate_for(GlFamilyProfile::Desktop { major: 4, minor: 6 }),
            Err(GlShaderValidationError::DialectProfileMismatch)
        );

        let glsl_400 = source(
            GlShaderStage::Vertex,
            GlShaderDialect::Desktop { version: 400 },
        );
        // A GL 4.6 driver accepts a GLSL 4.00 asset. Discovery reports the
        // ceiling; it must not force every asset to declare that ceiling.
        assert!(
            glsl_400
                .validate_for(GlFamilyProfile::Desktop { major: 4, minor: 6 })
                .is_ok()
        );
        assert_eq!(
            glsl_400.validate_for(GlFamilyProfile::Desktop { major: 3, minor: 3 }),
            Err(GlShaderValidationError::DialectProfileMismatch)
        );
        let glsl_470 = source(
            GlShaderStage::Vertex,
            GlShaderDialect::Desktop { version: 470 },
        );
        assert_eq!(
            glsl_470.validate_for(GlFamilyProfile::Desktop { major: 4, minor: 6 }),
            Err(GlShaderValidationError::DialectProfileMismatch)
        );

        let es31 = source(
            GlShaderStage::Compute,
            GlShaderDialect::Embedded { version: 310 },
        );
        assert!(
            es31.validate_for(GlFamilyProfile::Embedded { major: 3, minor: 1 })
                .is_ok()
        );
        assert_eq!(
            es31.validate_for(GlFamilyProfile::Embedded { major: 3, minor: 0 }),
            Err(GlShaderValidationError::DialectProfileMismatch)
        );
    }

    #[test]
    fn rejects_empty_lowered_entry_point() {
        let mut shader = source(
            GlShaderStage::Fragment,
            GlShaderDialect::Desktop { version: 430 },
        );
        shader.entry_point.clear();
        assert_eq!(
            shader.validate_for(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
            Err(GlShaderValidationError::EmptyEntryPoint)
        );
    }
    #[test]
    fn program_kind_routes_stage_validation() {
        let raster = |vertex: GlShaderSource, fragment: GlShaderSource| GlProgramDescriptor {
            kind: GlProgramKind::Raster { vertex, fragment },
            layout: GlPipelineLayout { bindings: vec![] },
            debug_name: None,
        };
        let desktop = GlFamilyProfile::Desktop { major: 4, minor: 3 };
        assert!(
            raster(
                source(
                    GlShaderStage::Vertex,
                    GlShaderDialect::Desktop { version: 430 }
                ),
                source(
                    GlShaderStage::Fragment,
                    GlShaderDialect::Desktop { version: 430 }
                ),
            )
            .validate_for(desktop)
            .is_ok()
        );
        assert_eq!(
            raster(
                source(
                    GlShaderStage::Compute,
                    GlShaderDialect::Desktop { version: 430 }
                ),
                source(
                    GlShaderStage::Fragment,
                    GlShaderDialect::Desktop { version: 430 }
                ),
            )
            .validate(),
            Err(GlShaderValidationError::WrongStage)
        );
        let mut compute = GlProgramDescriptor {
            kind: GlProgramKind::Compute {
                shader: source(
                    GlShaderStage::Vertex,
                    GlShaderDialect::Desktop { version: 430 },
                ),
            },
            layout: GlPipelineLayout { bindings: vec![] },
            debug_name: None,
        };
        assert_eq!(compute.validate(), Err(GlShaderValidationError::WrongStage));
        if let GlProgramKind::Compute { shader } = &mut compute.kind {
            shader.stage = GlShaderStage::Compute;
            shader.dialect = GlShaderDialect::Desktop { version: 430 };
        }
        assert!(compute.validate_for(desktop).is_ok());
    }
    #[test]
    fn rejects_duplicate_executable_assignments() {
        let a = GlBindingLocation {
            group: 0,
            binding: 0,
        };
        let b = GlBindingLocation {
            group: 0,
            binding: 1,
        };
        let layout = GlPipelineLayout {
            bindings: vec![
                GlLogicalBinding {
                    element: None,
                    names: vec!["a".into()],
                    location: a,
                    kind: GlShaderResourceKind::UniformBuffer,
                    array_count: 1,
                    fixed_array: false,
                    pair: None,
                    image_unit: None,
                    sampled_type: None,
                    image_type: None,
                },
                GlLogicalBinding {
                    element: None,
                    names: vec!["b".into()],
                    location: b,
                    kind: GlShaderResourceKind::UniformBuffer,
                    array_count: 1,
                    fixed_array: false,
                    pair: None,
                    image_unit: None,
                    sampled_type: None,
                    image_type: None,
                },
            ],
        };
        let reflection = GlProgramReflection {
            vertex_inputs: vec![],
            fragment_outputs: vec![],
            assignments: vec![
                GlExecutableBindingAssignment {
                    element: None,
                    logical: a,
                    executable: GlExecutableBindingLocation::UniformBlock(0),
                },
                GlExecutableBindingAssignment {
                    element: None,
                    logical: b,
                    executable: GlExecutableBindingLocation::UniformBlock(0),
                },
            ],
        };
        assert_eq!(
            reflection.validate_against(&layout),
            Err(GlShaderValidationError::DuplicateExecutableAssignment)
        );
    }

    #[test]
    fn storage_image_assignment_matches_its_explicit_glsl_unit() {
        let location = GlBindingLocation {
            group: 0,
            binding: 1,
        };
        let layout = GlPipelineLayout {
            bindings: vec![GlLogicalBinding {
                element: None,
                names: vec!["output_image".into()],
                location,
                kind: GlShaderResourceKind::StorageImage(GlStorageImageAccess::WriteOnly),
                array_count: 1,
                fixed_array: false,
                pair: None,
                image_unit: Some(0),
                sampled_type: None,
                image_type: None,
            }],
        };
        let make_reflection = |unit| GlProgramReflection {
            vertex_inputs: vec![],
            fragment_outputs: vec![],
            assignments: vec![GlExecutableBindingAssignment {
                element: None,
                logical: location,
                executable: GlExecutableBindingLocation::ImageUnit(unit),
            }],
        };

        assert!(make_reflection(0).validate_against(&layout).is_ok());
        assert_eq!(
            make_reflection(1).validate_against(&layout),
            Err(GlShaderValidationError::StorageImageUnitMismatch)
        );
        let mut missing = layout;
        missing.bindings[0].image_unit = None;
        // Direct GLSL gets the image unit from linked uniform reflection.
        assert!(make_reflection(0).validate_against(&missing).is_ok());
    }
