//! Checks mesh and amplification limits carried in native DXIL metadata.

use fluxel_rhi_core::api::{Error, MeshShaderLimits, PrimitiveTopology, Result, ShaderStage};

use super::dxil_container::{malformed, parts, shader_model, word};

// Additive layouts from DXC's DxilPipelineStateValidation.h. Info0 stores
// mesh/task output facts. Info1 adds topology, and Info2 adds numthreads.
const INFO0_BYTES: usize = 24;
const MESH_TOPOLOGY_OFFSET: usize = 27;
const NUMTHREADS_OFFSET: usize = 36;
const INFO2_BYTES: usize = 48;

pub(in crate::backend) fn validate(
    bytes: &[u8],
    stage: ShaderStage,
    declared_threads: Option<[u32; 3]>,
    topology: PrimitiveTopology,
    maximum_shader_model: u16,
    limits: MeshShaderLimits,
) -> Result<u32> {
    let (program, validation) = parts(bytes, *b"PSV0")?;
    if !(65..=maximum_shader_model).contains(&shader_model(word(program, 0)?)) {
        return Err(Error::Unsupported("DX12 mesh/task shader model"));
    }
    let validation = validation.ok_or(Error::Unsupported("DX12 mesh/task PSV0 metadata"))?;
    let size = word(validation, 0)? as usize;
    let runtime = validation
        .get(4..)
        .and_then(|data| data.get(..size))
        .ok_or_else(malformed)?;
    if runtime.len() < INFO0_BYTES {
        return Err(malformed());
    }
    let invocations = match stage {
        ShaderStage::Mesh => limits.max_mesh_workgroup_invocations,
        ShaderStage::Task => limits.max_task_workgroup_invocations,
        _ => return Err(Error::InvalidState),
    };
    if let Some(threads) = declared_threads {
        validate_threads(threads, invocations)?;
    }
    if runtime.len() >= INFO2_BYTES {
        let threads = [
            word(runtime, NUMTHREADS_OFFSET)?,
            word(runtime, NUMTHREADS_OFFSET + 4)?,
            word(runtime, NUMTHREADS_OFFSET + 8)?,
        ];
        validate_threads(threads, invocations)?;
        if declared_threads.is_some_and(|declared| declared != threads) {
            return Err(Error::Shader(
                "DX12 mesh/task numthreads differs from its declaration".into(),
            ));
        }
    }
    let payload = if stage == ShaderStage::Mesh {
        let output = word(runtime, 12)?;
        if (output & 0xffff) > limits.max_mesh_output_vertices
            || (output >> 16) > limits.max_mesh_output_primitives
        {
            return Err(Error::InvalidRange);
        }
        if let Some(&encoded) = runtime.get(MESH_TOPOLOGY_OFFSET) {
            let expected = match topology {
                PrimitiveTopology::LineList => 1,
                PrimitiveTopology::TriangleList => 2,
                _ => return Err(Error::Unsupported("DX12 mesh output topology")),
            };
            if encoded != expected {
                return Err(Error::Shader(
                    "DX12 mesh topology differs from its pipeline".into(),
                ));
            }
        }
        word(runtime, 8)?
    } else {
        word(runtime, 0)?
    };
    if payload > limits.max_task_payload_bytes {
        return Err(Error::InvalidRange);
    }
    Ok(payload)
}

fn validate_threads(threads: [u32; 3], maximum: u32) -> Result<()> {
    let product = threads.into_iter().try_fold(1u32, u32::checked_mul);
    if product.is_none_or(|count| count == 0 || count > maximum) {
        return Err(Error::InvalidRange);
    }
    Ok(())
}
