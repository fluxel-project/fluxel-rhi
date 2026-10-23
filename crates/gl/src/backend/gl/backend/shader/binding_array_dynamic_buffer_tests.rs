use super::*;

#[test]
fn uniform_array_dynamic_load_respects_separate_limit() {
    let source = r#"struct Data { value: vec4u }
@group(0) @binding(0) var<uniform> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1) fn main() { output[0] = buffers[selected].value.x; }"#;
    let mut rejected = naga::front::wgsl::parse_str(source).unwrap();
    assert!(matches!(
        lower_uniform(&mut rejected, &mut HashMap::new(), 1),
        Err(Error::Unsupported(_))
    ));
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    lower_uniform(&mut module, &mut aliases, 2).unwrap();
    assert_eq!(aliases.len(), 2);
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(verify_glsl(&module, &info).contains("switch"));
}

fn verify_glsl(module: &naga::Module, info: &naga::valid::ModuleInfo) -> String {
    let options = naga::back::glsl::Options {
        version: naga::back::glsl::Version::Desktop(450),
        writer_flags: naga::back::glsl::WriterFlags::empty(),
        binding_map: Default::default(),
        zero_initialize_workgroup_memory: true,
    };
    let pipeline = naga::back::glsl::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".into(),
        multiview: None,
    };
    let mut output = String::new();
    naga::back::glsl::Writer::new(
        &mut output,
        module,
        info,
        &options,
        &pipeline,
        Default::default(),
    )
    .unwrap()
    .write()
    .unwrap();
    output
}

#[test]
fn store_in_helper_and_loop() {
    let source = r#"struct Data { values: array<u32, 4> }
@group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
fn write(i: u32) { for (var j = 0u; j < 4u; j++) { buffers[i].values[j] = j; } }
@compute @workgroup_size(1) fn main() { write(selected); }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    lower(&mut module, &mut aliases, 2).unwrap();
    assert_eq!(aliases.len(), 2);
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let options = naga::back::glsl::Options {
        version: naga::back::glsl::Version::Desktop(450),
        writer_flags: naga::back::glsl::WriterFlags::empty(),
        binding_map: Default::default(),
        zero_initialize_workgroup_memory: true,
    };
    let pipeline = naga::back::glsl::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".into(),
        multiview: None,
    };
    let mut output = String::new();
    naga::back::glsl::Writer::new(
        &mut output,
        &module,
        &info,
        &options,
        &pipeline,
        Default::default(),
    )
    .unwrap()
    .write()
    .unwrap();
    assert!(output.contains("switch"), "{output}");
}

#[test]
fn load_and_store() {
    let source = r#"struct Data { values: array<u32, 4> }
@group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@compute @workgroup_size(1) fn main() { buffers[selected].values[1] = buffers[selected].values[0] + 1u; }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    lower(&mut module, &mut aliases, 2).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(verify_glsl(&module, &info).contains("switch"));
}

#[test]
fn local_initializer_survives_remapping() {
    let source = r#"struct Data { value: u32 }
@group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@compute @workgroup_size(1) fn main() { var x = 11u; x += buffers[selected].value; buffers[selected].value = x; }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    assert!(
        module.entry_points[0]
            .function
            .local_variables
            .iter()
            .any(|(_, local)| local.init.is_some())
    );
    lower(&mut module, &mut HashMap::new(), 2).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(verify_glsl(&module, &info).contains("11u"));
}

#[test]
fn atomic_result() {
    let source = r#"struct Data { value: atomic<u32> }
@group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: u32;
@compute @workgroup_size(1) fn main() { output = atomicAdd(&buffers[selected].value, 1u); }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    lower(&mut module, &mut aliases, 2).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(verify_glsl(&module, &info).contains("atomicAdd"));
}

#[test]
fn array_length() {
    let source = r#"struct Data { values: array<u32> }
@group(0) @binding(0) var<storage, read> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: u32;
@compute @workgroup_size(1) fn main() { output = arrayLength(&buffers[selected].values); }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    lower(&mut module, &mut HashMap::new(), 2).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(verify_glsl(&module, &info).contains(".length()"));
}

#[test]
fn multiple_atomic_operations() {
    let source = r#"struct Data { value: atomic<u32> }
@group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: u32;
@compute @workgroup_size(1) fn main() {
    let first = atomicAdd(&buffers[selected].value, 1u);
    let second = atomicExchange(&buffers[selected].value, 3u);
    output = first + second;
}"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut aliases = HashMap::new();
    lower(&mut module, &mut aliases, 2).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let glsl = verify_glsl(&module, &info);
    assert!(glsl.contains("atomicAdd") && glsl.contains("atomicExchange"));
}

#[test]
fn shared_atomic_pointer() {
    let source = r#"struct Data { value: atomic<u32> }
@group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: u32;
@compute @workgroup_size(1) fn main() { let p = &buffers[selected].value; let a = atomicAdd(p, 1u); let b = atomicExchange(p, 3u); output = a + b; }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    lower(&mut module, &mut HashMap::new(), 2).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(verify_glsl(&module, &info).contains("atomicAdd"));
}

#[test]
fn shared_atomic_pointer_compare_exchange() {
    let source = r#"struct Data { value: atomic<u32> }
@group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: u32;
@compute @workgroup_size(1) fn main() { let p = &buffers[selected].value; let a = atomicAdd(p, 1u); let b = atomicCompareExchangeWeak(p, 2u, 3u); output = a + b.old_value; }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    lower(&mut module, &mut HashMap::new(), 2).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(verify_glsl(&module, &info).contains("atomicCompSwap"));
}

#[test]
fn shared_root_atomic_different_scalar_types() {
    let source = r#"struct Data { a: atomic<u32>, b: atomic<i32> }
@group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: u32;
@compute @workgroup_size(1) fn main() {
    let p = &buffers[selected];
    let pa = &(*p).a;
    let pb = &(*p).b;
    let a = atomicAdd(pa, 1u);
    let b = atomicCompareExchangeWeak(pa, 2u, 3u);
    let c = atomicAdd(pb, 1i);
    let d = atomicCompareExchangeWeak(pb, 2i, 3i);
    output = a + b.old_value + u32(c + d.old_value);
}"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    lower(&mut module, &mut HashMap::new(), 2).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(verify_glsl(&module, &info).contains("atomicCompSwap"));
}

#[test]
fn pointer_escape_is_rejected() {
    let source = r#"struct Data { value: u32 }
@group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
fn write(p: ptr<storage, u32, read_write>) { *p = 9u; }
@compute @workgroup_size(1) fn main() { write(&buffers[selected].value); }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let error = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_err();
    assert!(format!("{error:?}").contains("InvalidArgumentPointerSpace"));
    let result = lower(&mut module, &mut HashMap::new(), 2);
    assert!(matches!(result, Err(Error::Unsupported(_))), "{result:?}");
}

#[test]
fn root_pointer_load() {
    let source = r#"struct Data { value: u32 }
@group(0) @binding(0) var<storage, read> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: u32;
@compute @workgroup_size(1) fn main() { let data = buffers[selected]; output = data.value; }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    lower(&mut module, &mut HashMap::new(), 2).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(verify_glsl(&module, &info).contains("switch"));
}

#[test]
fn shared_root_and_member_load() {
    let source = r#"struct Data { value: u32 }
@group(0) @binding(0) var<storage, read> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: u32;
@compute @workgroup_size(1) fn main() { let p = &buffers[selected]; let data = *p; output = data.value + (*p).value; }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    lower(&mut module, &mut HashMap::new(), 2).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(verify_glsl(&module, &info).contains("switch"));
}

#[test]
fn shared_global_multiple_root_loads() {
    let source = r#"struct Data { value: u32 }
@group(0) @binding(0) var<storage, read> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: u32;
@compute @workgroup_size(1) fn main() { let a = buffers[selected]; let b = buffers[1u - selected]; output = a.value + b.value; }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let f = &mut module.entry_points[0].function;
    let roots: Vec<_> = f.expressions.iter().filter_map(|(h,e)| match *e {
        Expression::Access { base, .. } if matches!(f.expressions[base], Expression::GlobalVariable(g) if g.index() == 0) => Some((h,base)),
        _ => None,
    }).collect();
    assert_eq!(roots.len(), 2);
    if let Expression::Access { base, .. } = &mut f.expressions[roots[1].0] {
        *base = roots[0].1;
    }
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    lower(&mut module, &mut HashMap::new(), 2).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(verify_glsl(&module, &info).contains("switch"));
}

#[test]
fn shared_global_static_and_dynamic() {
    let source = r#"struct Data { value: u32 }
@group(0) @binding(0) var<storage, read> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: u32;
@compute @workgroup_size(1) fn main() { let a = buffers[selected]; let b = buffers[1]; output = a.value + b.value; }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let f = &mut module.entry_points[0].function;
    let roots: Vec<_> = f.expressions.iter().filter_map(|(h,e)| match *e {
        Expression::Access { base, .. } | Expression::AccessIndex { base, .. } if matches!(f.expressions[base], Expression::GlobalVariable(g) if g.index() == 0) => Some((h,base)),
        _ => None,
    }).collect();
    assert_eq!(roots.len(), 2);
    match &mut f.expressions[roots[1].0] {
        Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
            *base = roots[0].1
        }
        _ => unreachable!(),
    }
    let mut aliases = HashMap::new();
    super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
    lower(&mut module, &mut aliases, 2).unwrap();
    assert_eq!(aliases.len(), 2);
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(verify_glsl(&module, &info).contains("switch"));
}

#[test]
fn shared_pointer_load_and_store() {
    let source = r#"struct Data { value: u32 }
@group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: u32;
@compute @workgroup_size(1) fn main() { let p = &buffers[selected].value; output = *p; *p = 7u; }"#;
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    lower(&mut module, &mut HashMap::new(), 2).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(verify_glsl(&module, &info).contains("switch"));
}

#[test]
fn storage_pointer_helper_multiple_paths_are_not_validated_by_naga() {
    let source = r#"struct Data { values: array<u32, 4> }
@group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 2>;
@group(0) @binding(1) var<uniform> indices: vec4u;
fn copy_values(dst: ptr<storage, u32, read_write>, src: ptr<storage, u32, read_write>) { *dst = *src; }
@compute @workgroup_size(1) fn main() { copy_values(&buffers[indices.x].values[indices.y], &buffers[indices.z].values[indices.w]); }"#;
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let error = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_err();
    assert!(format!("{error:?}").contains("InvalidArgumentPointerSpace"));
}

#[test]
fn unrestricted_pointer_parameters_extension_is_unimplemented_by_naga() {
    let source =
        "requires unrestricted_pointer_parameters; @compute @workgroup_size(1) fn main() {}";
    let error = naga::front::wgsl::parse_str(source).unwrap_err();
    assert!(format!("{error:?}").contains("unrestricted_pointer_parameters"));
}
