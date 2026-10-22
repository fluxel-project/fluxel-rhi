//! Draw-based attachment clears restricted to a Metal render area.
//!
//! Metal load-action clears cover an entire attachment. A partial render area
//! instead loads the attachment and runs this draw before user commands. Write
//! masks preserve all non-cleared MRT slots; depth and stencil state writes
//! only the aspects whose load operation is Clear.

#[cfg(target_vendor = "apple")]
pub(super) fn clear_area(
    render: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
    desc: &fluxel_rhi_core::api::RasterPassDesc<'_, super::super::super::Metal>,
    samples: usize,
) -> fluxel_rhi_core::api::Result<()> {
    use fluxel_rhi_core::api::{self, ClearColor, Error, LoadOp};
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLColorWriteMask, MTLCommandEncoder as _, MTLCompareFunction, MTLCullMode,
        MTLDepthStencilDescriptor, MTLDevice as _, MTLLibrary as _, MTLPixelFormat,
        MTLPrimitiveType, MTLRenderCommandEncoder as _, MTLRenderPipelineDescriptor,
        MTLScissorRect, MTLStencilDescriptor, MTLStencilOperation, MTLTexture as _,
        MTLTriangleFillMode, MTLViewport,
    };
    let depth_clear = desc.depth_stencil.as_ref().and_then(|d| match d.depth {
        Some((LoadOp::Clear(value), _)) => Some(value),
        _ => None,
    });
    let stencil_clear = desc.depth_stencil.as_ref().and_then(|d| match d.stencil {
        Some((LoadOp::Clear(value), _)) => Some(value),
        _ => None,
    });
    let has_color_clear = desc
        .colors
        .iter()
        .flatten()
        .any(|c| matches!(c.load, LoadOp::Clear(_)));
    if !has_color_clear && depth_clear.is_none() && stencil_clear.is_none() {
        return Ok(());
    }
    if desc.colors.len() > 8
        || samples == 0
        || desc.render_area.width == 0
        || desc.render_area.height == 0
    {
        return Err(Error::InvalidRange);
    }

    let mut source = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
    let depth = depth_clear.unwrap_or(0.0);
    let layers = if let Some(mask) = desc.multiview_mask {
        if mask == 0 {
            return Err(Error::InvalidRange);
        }
        let indices: Vec<u32> = (0..32).filter(|i| mask & (1 << i) != 0).collect();
        source.push_str("struct ClearVertex { float4 position [[position]]; uint layer [[render_target_array_index]]; };\n");
        source.push_str("vertex ClearVertex clear_vertex(uint i [[vertex_id]], uint instance [[instance_id]]) { float2 p = i == 0 ? float2(-1.0,-1.0) : (i == 1 ? float2(3.0,-1.0) : float2(-1.0,3.0)); ClearVertex o; ");
        source.push_str(&format!(
            "o.position = float4(p, as_type<float>({}u), 1.0); ",
            depth.to_bits()
        ));
        source.push_str("switch(instance) { ");
        for (instance, layer) in indices.iter().enumerate() {
            source.push_str(&format!("case {instance}u: o.layer = {layer}u; break; "));
        }
        source.push_str("default: o.layer = 0u; break; } return o; }\n");
        indices.len()
    } else {
        source.push_str(&format!("vertex float4 clear_vertex(uint i [[vertex_id]]) {{ float2 p = i == 0 ? float2(-1.0,-1.0) : (i == 1 ? float2(3.0,-1.0) : float2(-1.0,3.0)); return float4(p, as_type<float>({}u), 1.0); }}\n", depth.to_bits()));
        1
    };
    if has_color_clear {
        source.push_str("struct ClearOutputs {\n");
        for (index, attachment) in desc.colors.iter().enumerate() {
            if let Some(attachment) = attachment {
                if let LoadOp::Clear(color) = attachment.load {
                    let ty = match color {
                        ClearColor::Float(_) => "float4",
                        ClearColor::Uint(_) => "uint4",
                        ClearColor::Sint(_) => "int4",
                    };
                    source.push_str(&format!("  {ty} c{index} [[color({index})]];\n"));
                }
            }
        }
        source.push_str("};\nfragment ClearOutputs clear_fragment() { ClearOutputs o;\n");
        for (index, attachment) in desc.colors.iter().enumerate() {
            if let Some(attachment) = attachment {
                if let LoadOp::Clear(color) = attachment.load {
                    let bits = match color {
                        ClearColor::Float(values) => values.map(f32::to_bits),
                        ClearColor::Uint(values) => values,
                        ClearColor::Sint(values) => values.map(|v| v as u32),
                    };
                    let cast = match color {
                        ClearColor::Float(_) => "float4",
                        ClearColor::Uint(_) => "uint4",
                        ClearColor::Sint(_) => "int4",
                    };
                    source.push_str(&format!(
                        "  o.c{index} = as_type<{cast}>(uint4({}u,{}u,{}u,{}u));\n",
                        bits[0], bits[1], bits[2], bits[3]
                    ));
                }
            }
        }
        source.push_str("return o; }\n");
    } else {
        source.push_str("fragment void clear_fragment() {}\n");
    }
    let device = render.device();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(&source), None)
        .map_err(|error| Error::Shader(format!("Metal render area clear shader: {error:?}")))?;
    let vertex = library
        .newFunctionWithName(&NSString::from_str("clear_vertex"))
        .ok_or_else(|| Error::Shader("Metal clear vertex function missing".into()))?;
    let fragment = library
        .newFunctionWithName(&NSString::from_str("clear_fragment"))
        .ok_or_else(|| Error::Shader("Metal clear fragment function missing".into()))?;
    let pipeline_desc = MTLRenderPipelineDescriptor::new();
    pipeline_desc.setVertexFunction(Some(&vertex));
    pipeline_desc.setFragmentFunction(Some(&fragment));
    pipeline_desc.setRasterSampleCount(samples);
    let mut extent = None;
    for (index, attachment) in desc.colors.iter().enumerate() {
        let Some(attachment) = attachment else {
            continue;
        };
        // SAFETY: Core validated device ownership and live surface leases.
        let texture: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture> =
            match &attachment.view {
                api::AttachmentView::Texture(view) => &unsafe { view.native_ref() }.raw,
                api::AttachmentView::Surface(image) => &unsafe { image.native_ref() }.texture,
            };
        extent.get_or_insert((texture.width(), texture.height()));
        // SAFETY: the pass is limited to Metal's eight color attachment slots.
        let color = unsafe {
            pipeline_desc
                .colorAttachments()
                .objectAtIndexedSubscript(index)
        };
        color.setPixelFormat(texture.pixelFormat());
        color.setBlendingEnabled(false);
        color.setWriteMask(if matches!(attachment.load, LoadOp::Clear(_)) {
            MTLColorWriteMask::All
        } else {
            MTLColorWriteMask::None
        });
    }
    if let Some(depth) = &desc.depth_stencil {
        // SAFETY: Core checked the view device and this backend rejects surface depth.
        let texture = match &depth.view {
            api::AttachmentView::Texture(view) => &unsafe { view.native_ref() }.raw,
            api::AttachmentView::Surface(_) => return Err(Error::InvalidState),
        };
        extent.get_or_insert((texture.width(), texture.height()));
        let format = texture.pixelFormat();
        if format != MTLPixelFormat::Stencil8 {
            pipeline_desc.setDepthAttachmentPixelFormat(format);
        }
        if matches!(
            format,
            MTLPixelFormat::Depth32Float_Stencil8
                | MTLPixelFormat::Depth24Unorm_Stencil8
                | MTLPixelFormat::Stencil8
        ) {
            pipeline_desc.setStencilAttachmentPixelFormat(format);
        }
    }
    let (width, height) = extent.ok_or(Error::InvalidState)?;
    let area = desc.render_area;
    if u64::from(area.x) + u64::from(area.width) > width as u64
        || u64::from(area.y) + u64::from(area.height) > height as u64
    {
        return Err(Error::InvalidRange);
    }
    let pipeline = device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_desc)
        .map_err(|error| Error::Backend(format!("Metal render area clear pipeline: {error:?}")))?;
    let depth_desc = MTLDepthStencilDescriptor::new();
    depth_desc.setDepthCompareFunction(MTLCompareFunction::Always);
    depth_desc.setDepthWriteEnabled(depth_clear.is_some());
    if stencil_clear.is_some() {
        let stencil = MTLStencilDescriptor::new();
        stencil.setStencilCompareFunction(MTLCompareFunction::Always);
        stencil.setDepthStencilPassOperation(MTLStencilOperation::Replace);
        stencil.setWriteMask(255);
        depth_desc.setFrontFaceStencil(Some(&stencil));
        depth_desc.setBackFaceStencil(Some(&stencil));
    }
    let depth_state = device
        .newDepthStencilStateWithDescriptor(&depth_desc)
        .ok_or(Error::OutOfMemory)?;
    render.setRenderPipelineState(&pipeline);
    render.setDepthStencilState(Some(&depth_state));
    render.setCullMode(MTLCullMode::None);
    render.setTriangleFillMode(MTLTriangleFillMode::Fill);
    render.setStencilReferenceValue(stencil_clear.unwrap_or(0));
    render.setViewport(MTLViewport {
        originX: area.x as f64,
        originY: area.y as f64,
        width: area.width as f64,
        height: area.height as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    // SAFETY: scissor rectangle was checked against attachment extent, and the
    // three vertices are generated by the shader using the native vertex ID.
    unsafe {
        render.setScissorRect(MTLScissorRect {
            x: area.x as usize,
            y: area.y as usize,
            width: area.width as usize,
            height: area.height as usize,
        });
        render.drawPrimitives_vertexStart_vertexCount_instanceCount(
            MTLPrimitiveType::Triangle,
            0,
            3,
            layers,
        );
    }
    render.setViewport(MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: width as f64,
        height: height as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    render.setStencilReferenceValue(0);
    render.setDepthStencilState(None);
    Ok(())
}
