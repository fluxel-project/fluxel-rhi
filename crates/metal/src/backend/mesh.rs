//! Metal mesh pipeline creation and direct/indirect mesh dispatch.
//!
//! The native command buffer retains the pipeline and indirect argument buffer.
//! Threadgroup dimensions come from the corresponding shader entry declarations;
//! resource hazards and cross-queue synchronization remain the caller's responsibility.
use super::{Metal, MetalBuffer, MetalCommandEncoder, MetalDevice};
use fluxel_rhi_core::api::{self, MeshOps};
use std::future::{Ready, ready};

pub struct MetalMesh;
#[derive(Clone)]
pub struct MetalMeshPipeline {
    #[cfg(target_vendor = "apple")]
    raster: super::pipeline::MetalRasterPipeline,
    #[cfg(target_vendor = "apple")]
    threads: ([u32; 3], [u32; 3]),
}
impl MeshOps<Metal> for MetalMesh {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = Ready<api::Result<MetalMeshPipeline>>;
    fn create_pipeline<'a>(
        device: &'a MetalDevice,
        desc: &'a api::MeshPipelineDesc<Metal>,
    ) -> Self::PipelineFuture<'a> {
        ready(create(device, desc))
    }
    fn bind_pipeline(
        encoder: &mut MetalCommandEncoder,
        pipeline: &MetalMeshPipeline,
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use api::RasterBackend;
            use objc2_foundation::NSObjectProtocol as _;
            let render = encoder.render.as_ref().ok_or(api::Error::InvalidState)?;
            if !render.respondsToSelector(objc2::sel!(drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:))
                || !render.respondsToSelector(objc2::sel!(setMeshBuffer:offset:atIndex:))
                || !render.respondsToSelector(objc2::sel!(setMeshTexture:atIndex:))
                || !render.respondsToSelector(objc2::sel!(setMeshSamplerState:atIndex:)) {
                return Err(api::Error::Unsupported("Metal mesh encoder selectors"));
            }
            Metal::set_pipeline(encoder, &pipeline.raster)?;
            encoder.mesh_threads = Some(pipeline.threads);
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, pipeline);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn dispatch(encoder: &mut MetalCommandEncoder, groups: [u32; 3]) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLRenderCommandEncoder as _;
            let (object, mesh) = encoder.mesh_threads.ok_or(api::Error::InvalidState)?;
            let render = encoder.render.as_ref().ok_or(api::Error::InvalidState)?;
            if groups.contains(&0) {
                return Ok(());
            }
            render.drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                size(groups),
                size(object),
                size(mesh),
            );
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, groups);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn dispatch_indirect(
        encoder: &mut MetalCommandEncoder,
        args: &MetalBuffer,
        offset: u64,
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_foundation::NSObjectProtocol as _;
            use objc2_metal::{
                MTLCommandEncoder as _, MTLDevice as _, MTLGPUFamily, MTLRenderCommandEncoder as _,
            };
            let (object, mesh) = encoder.mesh_threads.ok_or(api::Error::InvalidState)?;
            if !valid_indirect(args.usage, args.size(), offset) {
                return Err(api::Error::InvalidRange);
            }
            let render = encoder.render.as_ref().ok_or(api::Error::InvalidState)?;
            let device = render.device();
            let selector = render.respondsToSelector(objc2::sel!(drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:));
            let family = device.respondsToSelector(objc2::sel!(supportsFamily:))
                && device.supportsFamily(MTLGPUFamily::Apple9);
            if !indirect_mesh_supported(family, selector) {
                return Err(api::Error::Unsupported(
                    "Metal indirect mesh requires Apple9",
                ));
            }
            // SAFETY: the argument range and alignment are checked; Core requires
            // the caller to synchronize and supply three u32 dispatch counts.
            // Metal's retained command buffer keeps the indirect buffer alive.
            unsafe {
                render.drawMeshThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(&args.raw, offset as usize, size(object), size(mesh));
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, args, offset);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
}
// Apple lists indirect mesh arguments for Apple9, with no Mac-family equivalent.
fn indirect_mesh_supported(apple9: bool, selector: bool) -> bool {
    apple9 && selector
}
/// Accept only MSL with an explicit `metal::mesh` triangle template. The
/// grammar is intentionally narrow: no preprocessing beyond metal_stdlib,
/// string literals, line splices, or alternate mesh type spellings. This makes
/// every mesh type in the translation unit visibly triangle-typed, including
/// types reached through `using` aliases. The native function lookup still
/// verifies that the requested entry is a mesh function.
fn proves_triangle_mesh(source: &str) -> bool {
    if !source.is_ascii()
        || source.contains('\\')
        || source.contains('?')
        || source.contains("%:")
        || source.contains('"')
        || source.contains('\'')
    {
        return false;
    }
    let mut lines = source.lines();
    if lines.next().map(str::trim) != Some("#include <metal_stdlib>") {
        return false;
    }
    let mut cleaned = String::new();
    for line in lines {
        if line.contains('#') {
            return false;
        }
        cleaned.push_str(line);
        cleaned.push('\n');
    }
    let b = cleaned.as_bytes();
    let mut tokens: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            i += 2;
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            i += 2;
            while i + 1 < b.len() && (b[i] != b'*' || b[i + 1] != b'/') {
                i += 1;
            }
            if i + 1 >= b.len() {
                return false;
            }
            i += 2;
            continue;
        }
        let start = i;
        if b[i].is_ascii_alphabetic() || b[i] == b'_' {
            i += 1;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
        } else if b[i].is_ascii_digit() {
            i += 1;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        } else {
            i += 1;
        }
        tokens.push(&cleaned[start..i]);
    }
    let identifier = |x: &str| {
        x.as_bytes()
            .first()
            .is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
            && x.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
    };
    let integer = |x: &str| !x.is_empty() && x.bytes().all(|c| c.is_ascii_digit());
    // Core has no object payload size or dynamic threadgroup-memory contract.
    // Reject those address spaces rather than leave native allocations implicit.
    if tokens
        .iter()
        .any(|token| matches!(*token, "object_data" | "threadgroup"))
    {
        return false;
    }
    let mut mesh_types = 0;
    for (n, token) in tokens.iter().enumerate() {
        if *token != "mesh" {
            continue;
        }
        if n >= 2 && tokens[n - 2..n] == ["[", "["] && tokens.get(n + 1..n + 3) == Some(&["]", "]"])
        {
            continue;
        }
        if n < 3 || tokens[n - 3..n] != ["metal", ":", ":"] {
            return false;
        }
        let Some(rest) = tokens.get(n + 1..n + 17) else {
            return false;
        };
        if rest[0] != "<"
            || !identifier(rest[1])
            || rest[2] != ","
            || !identifier(rest[3])
            || rest[4] != ","
            || !integer(rest[5])
            || rest[6] != ","
            || !integer(rest[7])
            || rest[8] != ","
            || rest[9..16] != ["metal", ":", ":", "topology", ":", ":", "triangle"]
            || tokens.get(n + 17) != Some(&">")
        {
            return false;
        }
        mesh_types += 1;
    }
    mesh_types > 0
}

fn valid_indirect(usage: api::BufferUsage, size: u64, offset: u64) -> bool {
    usage.contains(api::BufferUsage::INDIRECT)
        && offset % 4 == 0
        && offset.checked_add(12).is_some_and(|end| end <= size)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn triangle_proof_rejects_unprovable_source_forms() {
        let triangle = "#include <metal_stdlib>\nusing T = metal::mesh<V, P, 3, 1, metal::topology::triangle>; [[mesh]] void main(T out) {}";
        assert!(proves_triangle_mesh(triangle));
        assert!(!proves_triangle_mesh(&format!(
            "{triangle} object_data int* payload;"
        )));
        assert!(!proves_triangle_mesh(&format!(
            "{triangle} threadgroup int* shared;"
        )));
        assert!(!proves_triangle_mesh(
            &triangle.replace("using T", "%:define triangle line\nusing T")
        ));
        assert!(!proves_triangle_mesh(
            "#include <metal_stdlib>\n// metal::mesh<V,P,3,1,metal::topology::triangle>\n[[mesh]] void main(int x) {}"
        ));
        assert!(!proves_triangle_mesh(
            &triangle.replace("metal::topology::triangle", "metal::topology::point")
        ));
        assert!(!proves_triangle_mesh(
            &triangle.replace("topology::triangle", "topology::line")
        ));
        assert!(!proves_triangle_mesh(&triangle.replace(
            "#include <metal_stdlib>",
            "#define triangle line\n#include <metal_stdlib>"
        )));
        assert!(!proves_triangle_mesh(
            &triangle.replace("metal::mesh", "mesh")
        ));
        assert!(!proves_triangle_mesh(&format!(
            "{triangle} using L = metal::mesh<V, P, 3, 1, metal::topology::line>;"
        )));
        assert!(!proves_triangle_mesh(
            "/*\n#include <metal_stdlib>\n*/ metal::mesh<V,P,3,1,metal::topology::triangle> value;"
        ));
        assert!(!proves_triangle_mesh(
            &triangle.replace("metal::topology::triangle", "Other")
        ));
        assert!(!proves_triangle_mesh(
            &triangle.replace(" 3, 1,", " MAX, 1,")
        ));
        assert!(!proves_triangle_mesh(
            &triangle.replace("using T", "\\\nusing T")
        ));
        assert!(!proves_triangle_mesh(
            &triangle.replace("using T", "/* using T")
        ));
    }
    #[test]
    fn indirect_arguments_require_usage_alignment_and_complete_record() {
        let usage = api::BufferUsage::INDIRECT;
        assert!(valid_indirect(usage, 16, 4));
        assert!(!valid_indirect(api::BufferUsage::COPY_SRC, 16, 4));
        assert!(!valid_indirect(usage, 16, 3));
        assert!(!valid_indirect(usage, 15, 4));
        assert!(!valid_indirect(usage, u64::MAX, u64::MAX - 3));
    }

    #[test]
    fn indirect_mesh_requires_apple9_and_selector() {
        assert!(indirect_mesh_supported(true, true));
        assert!(!indirect_mesh_supported(false, true));
        assert!(!indirect_mesh_supported(true, false));
        assert!(!indirect_mesh_supported(false, false));
    }
}

#[cfg(target_vendor = "apple")]
fn size(v: [u32; 3]) -> objc2_metal::MTLSize {
    objc2_metal::MTLSize {
        width: v[0] as usize,
        height: v[1] as usize,
        depth: v[2] as usize,
    }
}

/// Reports the Metal 3 device and pipeline selector prerequisites.
/// Encoder selectors are checked separately when binding a render pipeline.
pub(super) fn mesh_supported(device: &MetalDevice) -> bool {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::NSObjectProtocol as _;
        use objc2_metal::{MTLDevice as _, MTLGPUFamily};
        let native = &device._state._device;
        native.respondsToSelector(objc2::sel!(supportsFamily:))
            && native.respondsToSelector(
                objc2::sel!(newRenderPipelineStateWithMeshDescriptor:options:reflection:error:),
            )
            && native.supportsFamily(MTLGPUFamily::Metal3)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = device;
        false
    }
}

fn create(
    device: &MetalDevice,
    desc: &api::MeshPipelineDesc<Metal>,
) -> api::Result<MetalMeshPipeline> {
    use super::pipeline as p;
    if desc.task.is_some()
        || !desc.mesh.constants.is_empty()
        || desc
            .fragment
            .as_ref()
            .is_none_or(|s| !s.constants.is_empty())
        || desc.cache.is_some()
        || !matches!(
            desc.primitive.topology,
            api::PrimitiveTopology::TriangleList
        )
        || desc.primitive.strip_index_format.is_some()
        || !matches!(desc.primitive.polygon_mode, api::PolygonMode::Fill)
        || desc.primitive.unclipped_depth
        || desc.primitive.conservative
        || desc
            .depth_stencil
            .is_some_and(|d| !p::supported_depth_state(d))
        || !p::supported_sample_count(desc.multisample.count)
        || !p::full_sample_mask(desc.multisample.mask)
        || desc.targets.len() != 1
        || desc.targets[0]
            .as_ref()
            .is_none_or(|t| t.write_mask & !15 != 0 || !p::supported_color_format(t.format))
    {
        return Err(api::Error::Unsupported("Metal mesh pipeline descriptor"));
    }
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLDepthStencilDescriptor, MTLDevice as _, MTLFunction as _, MTLFunctionType,
            MTLLibrary as _, MTLMeshRenderPipelineDescriptor, MTLPipelineOption,
            MTLStencilDescriptor,
        };
        let native = &device._state._device;
        if !mesh_supported(device) {
            return Err(api::Error::Unsupported(
                "Metal mesh requires Metal 3 family and selectors",
            ));
        }
        if !native.supportsTextureSampleCount(desc.multisample.count as usize) {
            return Err(api::Error::Unsupported("Metal mesh sample count"));
        }
        if desc
            .depth_stencil
            .is_some_and(|d| d.format == api::TextureFormat::Depth32FloatStencil8)
            && !super::texture::supports_depth32_stencil(device)
        {
            return Err(api::Error::Unsupported("Metal depth stencil format"));
        }
        let function = |stage: &api::PipelineStage<Metal>,
                        expected: api::ShaderStage,
                        native_type: MTLFunctionType|
         -> api::Result<_> {
            // SAFETY: Core verifies shader device identity before entering SPI.
            let shader = unsafe { stage.module.native_ref() };
            let entry = shader
                .entries
                .iter()
                .find(|e| e.name == stage.entry && e.stage == expected)
                .ok_or_else(|| api::Error::Shader("Metal mesh entry metadata missing".into()))?;
            let function = shader
                ._library
                .newFunctionWithName(&NSString::from_str(&stage.entry))
                .ok_or_else(|| api::Error::Shader("Metal mesh function missing".into()))?;
            if function.functionType() != native_type {
                return Err(api::Error::Shader(
                    "Metal mesh function type mismatch".into(),
                ));
            }
            Ok((function, entry.workgroup_size))
        };
        // SAFETY: Core has checked the mesh shader device identity. Only the
        // exact native source compiled into this library can establish topology.
        let mesh_shader = unsafe { desc.mesh.module.native_ref() };
        if !mesh_shader
            .mesh_source
            .as_deref()
            .is_some_and(proves_triangle_mesh)
        {
            return Err(api::Error::Unsupported(
                "Metal mesh shader topology cannot be verified",
            ));
        }
        let (mesh_fn, mesh_threads) =
            function(&desc.mesh, api::ShaderStage::Mesh, MTLFunctionType::Mesh)?;
        let mesh_threads =
            mesh_threads.ok_or_else(|| api::Error::Shader("Metal mesh threads missing".into()))?;
        let (object_fn, object_threads) = if let Some(stage) = &desc.task {
            let (function, threads) =
                function(stage, api::ShaderStage::Task, MTLFunctionType::Object)?;
            (
                Some(function),
                threads.ok_or_else(|| api::Error::Shader("Metal object threads missing".into()))?,
            )
        } else {
            (None, [1, 1, 1])
        };
        let (fragment, _) = function(
            desc.fragment.as_ref().unwrap(),
            api::ShaderStage::Fragment,
            MTLFunctionType::Fragment,
        )?;
        let max = native.maxThreadsPerThreadgroup();
        let validate = |v: [u32; 3]| -> api::Result<usize> {
            let product = v
                .iter()
                .try_fold(1usize, |n, x| n.checked_mul(*x as usize))
                .ok_or(api::Error::InvalidRange)?;
            if v.contains(&0)
                || v[0] as usize > max.width
                || v[1] as usize > max.height
                || v[2] as usize > max.depth
                || product > 1024
            {
                return Err(api::Error::InvalidRange);
            }
            Ok(product)
        };
        let mesh_count = validate(mesh_threads)?;
        let object_count = validate(object_threads)?;
        let descriptor = MTLMeshRenderPipelineDescriptor::new();
        // SAFETY: the device supports Metal 3 and function types were validated.
        unsafe {
            descriptor.setMeshFunction(Some(&mesh_fn));
            descriptor.setObjectFunction(object_fn.as_deref());
            descriptor.setFragmentFunction(Some(&fragment));
            descriptor.setRasterSampleCount(desc.multisample.count as usize);
        }
        descriptor.setMaxTotalThreadsPerMeshThreadgroup(mesh_count);
        if object_fn.is_some() {
            descriptor.setMaxTotalThreadsPerObjectThreadgroup(object_count);
        }
        descriptor.setAlphaToCoverageEnabled(desc.multisample.alpha_to_coverage_enabled);
        let depth_format = desc
            .depth_stencil
            .map(|d| {
                super::texture::metal_format(d.format)
                    .ok_or(api::Error::Unsupported("Metal mesh depth format"))
            })
            .transpose()?;
        let depth_state = if let Some(depth) = desc.depth_stencil {
            let format = depth_format.unwrap();
            descriptor.setDepthAttachmentPixelFormat(format);
            let state = MTLDepthStencilDescriptor::new();
            state.setDepthCompareFunction(p::compare_function(depth.depth_compare));
            state.setDepthWriteEnabled(depth.depth_write_enabled);
            if depth.format == api::TextureFormat::Depth32FloatStencil8 {
                descriptor.setStencilAttachmentPixelFormat(format);
                let front = MTLStencilDescriptor::new();
                p::configure_stencil_descriptor(
                    &front,
                    depth.stencil_front,
                    depth.stencil_read_mask,
                    depth.stencil_write_mask,
                );
                let back = MTLStencilDescriptor::new();
                p::configure_stencil_descriptor(
                    &back,
                    depth.stencil_back,
                    depth.stencil_read_mask,
                    depth.stencil_write_mask,
                );
                state.setFrontFaceStencil(Some(&front));
                state.setBackFaceStencil(Some(&back));
            }
            Some(
                native
                    .newDepthStencilStateWithDescriptor(&state)
                    .ok_or(api::Error::OutOfMemory)?,
            )
        } else {
            None
        };
        let target = desc.targets[0].as_ref().unwrap();
        let color_format = super::texture::metal_format(target.format)
            .ok_or(api::Error::Unsupported("Metal mesh color format"))?;
        // SAFETY: the descriptor contract accepts exactly one color target.
        let color = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        color.setPixelFormat(color_format);
        color.setWriteMask(p::write_mask(target.write_mask)?);
        if let Some(blend) = target.blend {
            color.setBlendingEnabled(true);
            color.setSourceRGBBlendFactor(p::blend_factor(blend.color.src_factor)?);
            color.setDestinationRGBBlendFactor(p::blend_factor(blend.color.dst_factor)?);
            color.setRgbBlendOperation(p::blend_operation(blend.color.operation));
            color.setSourceAlphaBlendFactor(p::blend_factor(blend.alpha.src_factor)?);
            color.setDestinationAlphaBlendFactor(p::blend_factor(blend.alpha.dst_factor)?);
            color.setAlphaBlendOperation(p::blend_operation(blend.alpha.operation));
        }
        let state = native
            .newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|e| api::Error::Backend(format!("Metal mesh pipeline: {e:?}")))?;
        // The actual native limits may be lower than the general device maximum.
        use objc2_metal::MTLRenderPipelineState as _;
        if mesh_count > state.maxTotalThreadsPerMeshThreadgroup()
            || object_fn.is_some() && object_count > state.maxTotalThreadsPerObjectThreadgroup()
        {
            return Err(api::Error::InvalidRange);
        }
        let raster = super::pipeline::MetalRasterPipeline {
            state,
            cull_mode: desc.primitive.cull_mode,
            front_face: desc.primitive.front_face,
            depth_state,
            depth_bias: desc.depth_stencil.map_or((0, 0.0, 0.0), |d| {
                (
                    d.depth_bias_constant,
                    d.depth_bias_slope_scale,
                    d.depth_bias_clamp,
                )
            }),
            signature: (desc.multisample.count as usize, color_format, depth_format),
        };
        Ok(MetalMeshPipeline {
            raster,
            threads: (object_threads, mesh_threads),
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = device;
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}
