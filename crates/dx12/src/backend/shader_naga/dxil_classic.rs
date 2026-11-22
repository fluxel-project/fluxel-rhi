//! Checks classic raster input linkage from the DXIL PSV runtime prefix.

use fluxel_rhi_core::api::{Error, PrimitiveTopology, Result};
use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_LINE, D3D_PRIMITIVE_LINE_ADJ, D3D_PRIMITIVE_POINT,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, D3D_PRIMITIVE_TRIANGLE, D3D_PRIMITIVE_TRIANGLE_ADJ,
};

use super::dxil_container::{malformed, parts, shader_model, word};

fn runtime_prefix(bytes: &[u8], maximum_model: u16) -> Result<&[u8]> {
    let (program, validation) = parts(bytes, *b"PSV0")?;
    if !(60..=maximum_model).contains(&shader_model(word(program, 0)?)) {
        return Err(Error::Unsupported(
            "DX12 additional raster stage shader model",
        ));
    }
    let validation = validation.ok_or_else(malformed)?;
    let size = word(validation, 0)? as usize;
    if size < 16 {
        return Err(malformed());
    }
    validation
        .get(4..)
        .and_then(|data| data.get(..size))
        .ok_or_else(malformed)
}

pub(in crate::backend) fn validate_classic_linkage(
    topology: PrimitiveTopology,
    control: &[u8],
    evaluation: &[u8],
    geometry: &[u8],
    maximum_model: u16,
    conservative: bool,
) -> Result<()> {
    use PrimitiveTopology::*;
    let geometry_input = match topology {
        PointList => D3D_PRIMITIVE_POINT,
        LineList | LineStrip => D3D_PRIMITIVE_LINE,
        TriangleList | TriangleStrip => D3D_PRIMITIVE_TRIANGLE,
        LineListAdjacency | LineStripAdjacency => D3D_PRIMITIVE_LINE_ADJ,
        TriangleListAdjacency | TriangleStripAdjacency => D3D_PRIMITIVE_TRIANGLE_ADJ,
        PatchList(points) => {
            // PSVRuntimeInfo0 contains HSInfo/DSInfo at the start of its union.
            let hull = runtime_prefix(control, maximum_model)?;
            let domain = runtime_prefix(evaluation, maximum_model)?;
            if word(hull, 0)? != points
                || word(hull, 4)? != word(domain, 0)?
                || word(hull, 8)? != word(domain, 8)?
            {
                return Err(Error::Shader(
                    "DX12 tessellation patch size or domain disagrees across stages".into(),
                ));
            }
            // PSV HSInfo.TessellatorOutputPrimitive follows DXIL's enum.
            match word(hull, 12)? {
                1 => D3D_PRIMITIVE_POINT,
                2 => D3D_PRIMITIVE_LINE,
                3 | 4 => D3D_PRIMITIVE_TRIANGLE,
                _ => return Err(malformed()),
            }
        }
    };
    if !geometry.is_empty() {
        let geometry = runtime_prefix(geometry, maximum_model)?;
        if word(geometry, 0)? != geometry_input.0 as u32 {
            return Err(Error::Shader(
                "DX12 geometry shader input disagrees with the assembled primitive".into(),
            ));
        }
        if conservative && word(geometry, 4)? != D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP.0 as u32 {
            return Err(Error::Unsupported(
                "DX12 conservative rasterization requires triangle output",
            ));
        }
    }
    Ok(())
}
