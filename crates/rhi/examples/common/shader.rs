//! Example-only WGSL artifact production for native backends.
//!
//! Vulkan accepts the source form through the RHI's Naga SPIR-V lowering. DX12
//! and GL advertise only their native forms, so examples must materialize those
//! forms before asking the portable device to create a shader module.

use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use fluxel_rhi::api::binding::{BindingKind, StorageAccess};
use fluxel_rhi::api::error::{RhiError, RhiErrorKind, RhiResult};
use fluxel_rhi::api::platform::BackendKind;
use fluxel_rhi::api::shader::{
    ArtifactHash, ArtifactProducerVersion, GlslProfile, ShaderAbiVersion, ShaderArtifact,
    ShaderCode, ShaderInterface, ShaderModule, ShaderRequirements, ShaderStage,
};

const DXC: &str = r"C:\VulkanSDK\1.4.357.0\Bin\dxc.exe";
static NEXT_TEMPORARY_ARTIFACT: AtomicU64 = AtomicU64::new(0);
static GL_TARGET: Mutex<Option<ExampleShaderTarget>> = Mutex::new(None);

/// The native source form selected by an example runner.
///
/// GL needs an explicit version because a `Device` records its backend family,
/// but deliberately does not expose the discovered GL context profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExampleShaderTarget {
    Vulkan,
    Dx12,
    /// Browser WebGPU consumes the authored WGSL directly.
    WebGpu,
    Glsl {
        version: u16,
    },
    GlslEs {
        version: u16,
    },
}

/// Registers the actual GL dialect observed when an example session opens.
/// Recovery replaces the value when it opens a new context.
pub fn set_gl_target(target: ExampleShaderTarget) {
    *GL_TARGET
        .lock()
        .unwrap_or_else(|poison| poison.into_inner()) = Some(target);
}

/// An example shader-production error with the source lowering or compiler
/// diagnostic preserved for the host runner.
#[derive(Debug)]
pub struct ShaderBuildError(String);

impl fmt::Display for ShaderBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ShaderBuildError {}

/// Produces one target-native artifact from canonical WGSL and its already
/// declared portable interface.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors ShaderArtifact::new so call sites keep the full artifact contract visible"
)]
pub fn artifact(
    target: ExampleShaderTarget,
    stage: ShaderStage,
    entry_point: impl Into<String>,
    wgsl: impl AsRef<str>,
    abi_version: ShaderAbiVersion,
    interface: ShaderInterface,
    requirements: ShaderRequirements,
    content_hash: ArtifactHash,
    producer_version: ArtifactProducerVersion,
) -> Result<ShaderArtifact, ShaderBuildError> {
    let entry_point = entry_point.into();
    let wgsl = wgsl.as_ref();
    let code = match target {
        ExampleShaderTarget::Vulkan | ExampleShaderTarget::WebGpu => {
            ShaderCode::Wgsl(Arc::from(wgsl))
        }
        ExampleShaderTarget::Dx12 => ShaderCode::Dxil(Arc::from(compile_dxil(
            stage,
            &entry_point,
            wgsl,
            &interface,
        )?)),
        ExampleShaderTarget::Glsl { version } => ShaderCode::Glsl {
            version,
            profile: GlslProfile::Core,
            source: Arc::from(compile_glsl(
                stage,
                &entry_point,
                wgsl,
                version,
                false,
                &interface,
            )?),
        },
        ExampleShaderTarget::GlslEs { version } => ShaderCode::GlslEs {
            version,
            source: Arc::from(compile_glsl(
                stage,
                &entry_point,
                wgsl,
                version,
                true,
                &interface,
            )?),
        },
    };
    Ok(ShaderArtifact::new(
        stage,
        entry_point,
        code,
        abi_version,
        interface,
        requirements,
        content_hash,
        producer_version,
    ))
}

/// Chooses the target where its form follows the backend family alone.
///
/// OpenGL is intentionally refused here: callers must use [`artifact`] with a
/// concrete GLSL dialect matching their discovered context.
pub fn artifact_for_device(
    device: &fluxel_rhi::api::platform::Device,
    stage: ShaderStage,
    entry_point: impl Into<String>,
    wgsl: impl AsRef<str>,
    abi_version: ShaderAbiVersion,
    interface: ShaderInterface,
    requirements: ShaderRequirements,
    content_hash: ArtifactHash,
    producer_version: ArtifactProducerVersion,
) -> Result<ShaderArtifact, ShaderBuildError> {
    let target = match device.backend() {
        BackendKind::Vulkan => ExampleShaderTarget::Vulkan,
        BackendKind::Dx12 => ExampleShaderTarget::Dx12,
        // WebGPU consumes WGSL directly.  Keeping it in this shared helper
        // lets browser examples use the same authored artifact as the native
        // runners without involving a desktop compiler.
        BackendKind::WebGpu => ExampleShaderTarget::WebGpu,
        BackendKind::OpenGl | BackendKind::WebGl2 => (*GL_TARGET
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()))
        .ok_or_else(|| {
            ShaderBuildError(
                "the example runner has not provided the discovered GL GLSL version".into(),
            )
        })?,
        other => {
            return Err(ShaderBuildError(format!(
                "no WGSL example lowering for {other:?}"
            )));
        }
    };
    artifact(
        target,
        stage,
        entry_point,
        wgsl,
        abi_version,
        interface,
        requirements,
        content_hash,
        producer_version,
    )
}

/// Creates a shader from an example-authored artifact, lowering its WGSL to
/// the code form accepted by the selected native backend first.
pub async fn create_shader(
    device: &fluxel_rhi::api::platform::Device,
    authored: &ShaderArtifact,
) -> RhiResult<ShaderModule> {
    let ShaderCode::Wgsl(source) = &authored.code else {
        return device.create_shader(authored).await;
    };
    let mut native = artifact_for_device(
        device,
        authored.stage,
        authored.entry_point.clone(),
        source.as_ref(),
        authored.abi_version,
        authored.interface.clone(),
        authored.requirements.clone(),
        authored.content_hash,
        authored.producer_version,
    )
    .map_err(|error| RhiError::new(RhiErrorKind::Unsupported, error.to_string()))?;
    native.label = authored.label.clone();
    device.create_shader(&native).await
}

fn compile_dxil(
    stage: ShaderStage,
    entry_point: &str,
    wgsl: &str,
    interface: &ShaderInterface,
) -> Result<Vec<u8>, ShaderBuildError> {
    let module = parse_and_validate(wgsl)?;
    let mut options = naga::back::hlsl::Options {
        shader_model: naga::back::hlsl::ShaderModel::V6_0,
        fake_missing_bindings: false,
        ..Default::default()
    };
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = global.binding else {
            continue;
        };
        let space = u8::try_from(binding.group).map_err(|_| {
            ShaderBuildError(format!(
                "DX12 HLSL lowering cannot represent bind group {} as a register space",
                binding.group
            ))
        })?;
        options.binding_map.insert(
            binding,
            naga::back::hlsl::BindTarget {
                space,
                register: binding.binding,
                binding_array_size: None,
                dynamic_storage_buffer_offsets_index: None,
                restrict_indexing: false,
            },
        );
    }
    let pipeline = naga::back::hlsl::PipelineOptions {
        entry_point: Some((naga_stage(stage)?, entry_point.to_owned())),
    };
    let mut hlsl = String::new();
    naga::back::hlsl::Writer::new(&mut hlsl, &options, &pipeline)
        .write(&module, &validate_module(&module)?, None)
        .map_err(|error| ShaderBuildError(format!("Naga could not emit HLSL: {error}")))?;
    let hlsl = patch_dx12_abi(hlsl, interface);
    dxc(stage, entry_point, &hlsl)
}

fn compile_glsl(
    stage: ShaderStage,
    entry_point: &str,
    wgsl: &str,
    version: u16,
    es: bool,
    interface: &ShaderInterface,
) -> Result<String, ShaderBuildError> {
    let module = parse_and_validate(wgsl)?;
    let options = naga::back::glsl::Options {
        version: if es {
            naga::back::glsl::Version::new_gles(version)
        } else {
            naga::back::glsl::Version::Desktop(version)
        },
        ..Default::default()
    };
    let pipeline = naga::back::glsl::PipelineOptions {
        shader_stage: naga_stage(stage)?,
        entry_point: entry_point.to_owned(),
        multiview: None,
    };
    let mut glsl = String::new();
    let reflection = naga::back::glsl::Writer::new(
        &mut glsl,
        &module,
        &validate_module(&module)?,
        &options,
        &pipeline,
        naga::proc::BoundsCheckPolicies::default(),
    )
    .map_err(|error| ShaderBuildError(format!("Naga could not initialize GLSL lowering: {error}")))?
    .write()
    .map_err(|error| ShaderBuildError(format!("Naga could not emit GLSL: {error}")))?;
    let manifest = glsl_manifest(&module, &reflection, interface)?;
    Ok(format!("{manifest}{glsl}"))
}

/// Reflects exact Naga output names into the example-only GL resource ABI.
/// A GL pipeline can then bind portable (group, slot) resources without
/// deriving driver names from WGSL spelling.
fn glsl_manifest(
    module: &naga::Module,
    reflection: &naga::back::glsl::ReflectionInfo,
    interface: &ShaderInterface,
) -> Result<String, ShaderBuildError> {
    use fluxel_rhi::api::binding::BindingKind;

    let mut lines = String::new();
    for resource in interface.resources() {
        let handle = module
            .global_variables
            .iter()
            .find_map(|(handle, variable)| {
                variable.binding.as_ref().filter(|binding| {
                    binding.group == resource.group.get() && binding.binding == resource.slot.get()
                })?;
                Some(handle)
            })
            .ok_or_else(|| {
                ShaderBuildError(format!(
                    "GLSL resource ({},{}) has no WGSL global binding",
                    resource.group.get(),
                    resource.slot.get()
                ))
            })?;
        let (kind, name, pair) = match &resource.kind {
            BindingKind::UniformBuffer { .. } => {
                ("uniform", reflection.uniforms.get(&handle), None)
            }
            BindingKind::StorageBuffer { .. } => {
                ("storage-buffer", reflection.uniforms.get(&handle), None)
            }
            BindingKind::SampledTexture { .. } | BindingKind::StorageTexture { .. } => {
                let (name, mapping) = reflection
                    .texture_mapping
                    .iter()
                    .find(|(_, mapping)| mapping.texture == handle)
                    .ok_or_else(|| {
                        ShaderBuildError("Naga did not reflect a GLSL texture".into())
                    })?;
                let pair = mapping.sampler.and_then(|sampler| {
                    module.global_variables[sampler]
                        .binding
                        .as_ref()
                        .map(|binding| (binding.group, binding.binding))
                });
                (
                    if matches!(&resource.kind, BindingKind::StorageTexture { .. }) {
                        "storage-texture"
                    } else {
                        "sampled-texture"
                    },
                    Some(name),
                    pair,
                )
            }
            BindingKind::Sampler { .. } => {
                let (name, mapping) = reflection
                    .texture_mapping
                    .iter()
                    .find(|(_, mapping)| mapping.sampler == Some(handle))
                    .ok_or_else(|| {
                        ShaderBuildError("Naga did not reflect a GLSL sampler".into())
                    })?;
                let pair = module.global_variables[mapping.texture]
                    .binding
                    .as_ref()
                    .map(|binding| (binding.group, binding.binding));
                ("sampler", Some(name), pair)
            }
            _ => {
                return Err(ShaderBuildError(format!(
                    "GLSL example ABI does not cover {:?}",
                    resource.kind
                )));
            }
        };
        let name = name.ok_or_else(|| {
            ShaderBuildError(format!(
                "Naga omitted GLSL reflection name for ({},{})",
                resource.group.get(),
                resource.slot.get()
            ))
        })?;
        if !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(ShaderBuildError(format!(
                "invalid GLSL reflection name {name:?}"
            )));
        }
        lines.push_str(&format!(
            "// fluxel-gl-abi-v1 group={} slot={} kind={} name={}",
            resource.group.get(),
            resource.slot.get(),
            kind,
            name
        ));
        if let Some((group, slot)) = pair {
            lines.push_str(&format!(" pair_group={group} pair_slot={slot}"));
        }
        lines.push('\n');
    }
    Ok(lines)
}

fn parse_and_validate(wgsl: &str) -> Result<naga::Module, ShaderBuildError> {
    naga::front::wgsl::parse_str(wgsl)
        .map_err(|error| ShaderBuildError(format!("Naga could not parse WGSL: {error}")))
}

fn validate_module(module: &naga::Module) -> Result<naga::valid::ModuleInfo, ShaderBuildError> {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|error| ShaderBuildError(format!("Naga rejected WGSL: {error}")))
}

fn naga_stage(stage: ShaderStage) -> Result<naga::ShaderStage, ShaderBuildError> {
    match stage {
        ShaderStage::Vertex => Ok(naga::ShaderStage::Vertex),
        ShaderStage::Fragment => Ok(naga::ShaderStage::Fragment),
        ShaderStage::Compute => Ok(naga::ShaderStage::Compute),
        other => Err(ShaderBuildError(format!(
            "Naga example lowering does not support {other:?}"
        ))),
    }
}

fn patch_dx12_abi(mut hlsl: String, interface: &ShaderInterface) -> String {
    // Naga calls user locations LOC<n>, whereas the DX12 raster lowering pins
    // ABI 1.0 vertex elements to LOCATION<n>.
    hlsl = hlsl.replace(": LOC", ": LOCATION");
    for resource in interface.resources() {
        if !matches!(
            resource.kind,
            BindingKind::StorageTexture {
                access: StorageAccess::ReadOnly,
                ..
            }
        ) {
            continue;
        }
        let register = if resource.group.get() == 0 {
            format!("register(u{})", resource.slot.get())
        } else {
            format!(
                "register(u{}, space{})",
                resource.slot.get(),
                resource.group.get()
            )
        };
        let srv_register = register.replacen("register(u", "register(t", 1);
        hlsl = hlsl
            .lines()
            .map(|line| {
                if line.contains(&register) {
                    line.replace("RWTexture", "Texture")
                        .replace(&register, &srv_register)
                } else {
                    line.to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
    }
    hlsl
}

fn dxc(stage: ShaderStage, entry_point: &str, hlsl: &str) -> Result<Vec<u8>, ShaderBuildError> {
    let temporary = temporary_path();
    fs::create_dir_all(&temporary).map_err(|error| {
        ShaderBuildError(format!(
            "could not create DXIL temporary directory: {error}"
        ))
    })?;
    let source = temporary.join("shader.hlsl");
    let output = temporary.join("shader.dxil");
    let write_result = fs::write(&source, hlsl);
    if let Err(error) = write_result {
        let _ = fs::remove_dir_all(&temporary);
        return Err(ShaderBuildError(format!(
            "could not write generated HLSL: {error}"
        )));
    }
    let profile = match stage {
        ShaderStage::Vertex => "vs_6_0",
        ShaderStage::Fragment => "ps_6_0",
        ShaderStage::Compute => "cs_6_0",
        _ => unreachable!("naga_stage already rejected this stage"),
    };
    let compilation = Command::new(DXC)
        .args(["-T", profile, "-E", entry_point, "-Fo"])
        .arg(&output)
        .arg(&source)
        .output()
        .map_err(|error| ShaderBuildError(format!("could not launch {DXC}: {error}")));
    let result = match compilation {
        Ok(compilation) if compilation.status.success() => fs::read(&output).map_err(|error| {
            ShaderBuildError(format!("dxc succeeded but produced no DXIL: {error}"))
        }),
        Ok(compilation) => Err(ShaderBuildError(format!(
            "dxc failed for {profile} {entry_point}: {}",
            String::from_utf8_lossy(&compilation.stderr)
        ))),
        Err(error) => Err(error),
    };
    let _ = fs::remove_dir_all(&temporary);
    result
}

fn temporary_path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "fluxel-rhi-example-dxil-{}-{}",
        std::process::id(),
        NEXT_TEMPORARY_ARTIFACT.fetch_add(1, Ordering::Relaxed),
    ))
}
