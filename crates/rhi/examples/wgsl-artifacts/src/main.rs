//! Emits one validated Naga SPIR-V module per WGSL entry point.
//!
//! The PowerShell driver owns toolchain orchestration, ABI reflection, and
//! staging. This narrow program owns WGSL parsing, validation, SPIR-V
//! generation, and opt-in native HLSL emission/capability lowering for shader artifacts.

use std::{env, fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let source = args
        .next()
        .ok_or("usage: fluxel-example-wgsl-artifacts <source.wgsl> <vertex|fragment|compute> <entry> <output.spv> [expected-vertex-locations] [hlsl|push=group,binding,register,space hlsl-output]")?;
    let stage = parse_stage(&args.next().ok_or("missing shader stage")?)?;
    let entry = args.next().ok_or("missing entry point")?;
    let output = args.next().ok_or("missing output path")?;
    let expected_input_locations = args
        .next()
        .filter(|value| value != "-")
        .map(|value| parse_locations(&value))
        .transpose()?;
    let hlsl_mode = args.next();
    let (push_target, dx12_hlsl) = match hlsl_mode.as_deref() {
        Some("hlsl") => (None, args.next()),
        Some(value) if value.starts_with("push=") => {
            (Some(parse_push_target(value)?), args.next())
        }
        Some(_) => return Err("HLSL mode must be `hlsl` or a `push=...` target".into()),
        None => (None, None),
    };
    if hlsl_mode.is_some() && dx12_hlsl.is_none() {
        return Err("missing native HLSL output path".into());
    }
    if args.next().is_some() {
        return Err("unexpected extra argument".into());
    }

    let source = fs::read_to_string(source)?;
    let module = naga::front::wgsl::parse_str(&source)?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)?;
    if !module
        .entry_points
        .iter()
        .any(|point| point.stage == stage && point.name == entry)
    {
        return Err(format!("WGSL entry `{entry}` is not a {stage:?} entry point").into());
    }
    if let Some(expected) = expected_input_locations.as_ref() {
        let point = module
            .entry_points
            .iter()
            .find(|point| point.stage == stage && point.name == entry)
            .expect("entry point was checked above");
        let mut actual = Vec::new();
        for argument in &point.function.arguments {
            if let Some(naga::Binding::Location { location, .. }) = argument.binding {
                actual.push(location);
            } else {
                collect_struct_locations(&module, argument.ty, &mut actual);
            }
        }
        actual.sort_unstable();
        if actual.as_slice() != expected.as_slice() {
            return Err(format!(
                "WGSL vertex locations differ: expected {expected:?}, got {actual:?}"
            )
            .into());
        }
    }

    let mut output_module = module.clone();
    if let Some(push) = push_target {
        // Lower only the native artifact copy. The WGSL source keeps a regular
        // binding for WebGPU, while native backends receive a push-constant
        // module with the same 32-byte payload and draw semantics.
        let mut found = false;
        for (_, global) in output_module.global_variables.iter_mut() {
            if global.binding == Some(push.binding) {
                global.binding = None;
                global.space = naga::AddressSpace::PushConstant;
                found = true;
            }
        }
        if !found {
            return Err(format!(
                "native push lowering expected WGSL group {} binding {}",
                push.binding.group, push.binding.binding
            )
            .into());
        }
    }
    let output_info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&output_module)?;

    let words = naga::back::spv::write_vec(
        &output_module,
        &output_info,
        &naga::back::spv::Options {
            // Vulkan 1.3 accepts SPIR-V 1.6, but 1.5 keeps the generated
            // artifacts consumable by the repository's existing Vulkan path.
            lang_version: (1, 5),
            // WebGPU, Vulkan, and DX12 share the public coordinate convention.
            // Do not ask Naga to apply a target-specific position transform.
            flags: naga::back::spv::WriterFlags::empty(),
            ..Default::default()
        },
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: stage,
            entry_point: entry.clone(),
        }),
    )?;
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let output = Path::new(&output);
    let parent = output.parent().ok_or("output has no parent directory")?;
    fs::create_dir_all(parent)?;
    fs::write(output, bytes)?;

    if let Some(output) = dx12_hlsl {
        let mut options = naga::back::hlsl::Options {
            shader_model: naga::back::hlsl::ShaderModel::V6_0,
            fake_missing_bindings: false,
            ..Default::default()
        };
        if let Some(push) = push_target {
            options.binding_map.insert(
                naga::ResourceBinding {
                    group: 0,
                    binding: 0,
                },
                naga::back::hlsl::BindTarget {
                    register: 0,
                    ..Default::default()
                },
            );
            options.push_constants_target = Some(naga::back::hlsl::BindTarget {
                register: push.register,
                space: push.space,
                ..Default::default()
            });
        }
        let pipeline = naga::back::hlsl::PipelineOptions {
            entry_point: Some((stage, entry.clone())),
        };
        let mut hlsl = String::new();
        let mut writer = naga::back::hlsl::Writer::new(&mut hlsl, &options, &pipeline);
        writer.write(&output_module, &output_info, None)?;
        fs::write(output, hlsl)?;
    }
    Ok(())
}

#[derive(Copy, Clone)]
struct PushTarget {
    binding: naga::ResourceBinding,
    register: u32,
    space: u8,
}

fn parse_push_target(value: &str) -> Result<PushTarget, Box<dyn std::error::Error>> {
    let fields = value
        .strip_prefix("push=")
        .ok_or("push target must use push=group,binding,register,space")?
        .split(',')
        .collect::<Vec<_>>();
    if fields.len() != 4 {
        return Err("push target must contain group,binding,register,space".into());
    }
    Ok(PushTarget {
        binding: naga::ResourceBinding {
            group: fields[0].parse()?,
            binding: fields[1].parse()?,
        },
        register: fields[2].parse()?,
        space: fields[3].parse()?,
    })
}

// WGSL commonly groups vertex attributes in an input struct. Naga stores those
// locations on the struct members rather than on the entry-point argument, so
// ABI validation must walk the type instead of inspecting only arguments.
fn collect_struct_locations(module: &naga::Module, ty: naga::Handle<naga::Type>, out: &mut Vec<u32>) {
    if let naga::TypeInner::Struct { members, .. } = &module.types[ty].inner {
        for member in members {
            if let Some(naga::Binding::Location { location, .. }) = member.binding {
                out.push(location);
            } else {
                collect_struct_locations(module, member.ty, out);
            }
        }
    }
}

fn parse_locations(value: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut locations = if value.is_empty() {
        Vec::new()
    } else {
        value
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<_>, _>>()?
    };
    locations.sort_unstable();
    locations.dedup();
    Ok(locations)
}

fn parse_stage(value: &str) -> Result<naga::ShaderStage, Box<dyn std::error::Error>> {
    match value {
        "vertex" => Ok(naga::ShaderStage::Vertex),
        "fragment" => Ok(naga::ShaderStage::Fragment),
        "compute" => Ok(naga::ShaderStage::Compute),
        _ => Err(format!("unsupported shader stage `{value}`").into()),
    }
}
