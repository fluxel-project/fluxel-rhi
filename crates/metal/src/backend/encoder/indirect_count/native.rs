//! Private native conversion stage for GPU-counted draw commands.

use std::ptr::NonNull;

use fluxel_rhi_core::api::{self, IndexFormat};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLArgumentEncoder as _, MTLBlitCommandEncoder as _, MTLBuffer, MTLCommandBuffer as _,
    MTLCommandEncoder as _, MTLComputeCommandEncoder as _, MTLComputePipelineState as _,
    MTLDevice as _, MTLFunction as _, MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor,
    MTLIndirectCommandType, MTLLibrary as _, MTLResource, MTLResourceOptions, MTLResourceUsage,
    MTLSize,
};

use crate::backend::MetalBuffer;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Icb = Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>;
/// Device-owned immutable native objects for indirect-count conversion.
///
/// The pipeline and its source library are immutable after construction. This
/// cache deliberately retains no Rust `MetalDevice`/`DeviceState` Arc: normal
/// native Metal ownership is bounded by the owning DeviceState field.
#[derive(Clone)]
pub(in crate::backend) struct ConverterCache {
    _library: Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>,
    function: Retained<ProtocolObject<dyn objc2_metal::MTLFunction>>,
    pipeline: Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>,
}

impl ConverterCache {
    pub(in crate::backend) fn new(
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    ) -> api::Result<Self> {
        use objc2::runtime::NSObjectProtocol as _;
        let source = NSString::from_str(include_str!("../indirect_count.metal"));
        let options = objc2_metal::MTLCompileOptions::new();
        options.setLanguageVersion(objc2_metal::MTLLanguageVersion::Version2_2);
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|e| api::Error::Shader(format!("Metal ICB conversion: {e:?}")))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("fluxel_generate_draws"))
            .ok_or_else(|| api::Error::Shader("Metal ICB conversion entry missing".into()))?;
        if !function.respondsToSelector(objc2::sel!(newArgumentEncoderWithBufferIndex:)) {
            return Err(api::Error::Unsupported("Metal ICB argument encoder"));
        }
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| api::Error::Backend(format!("Metal ICB pipeline: {e:?}")))?;
        Ok(Self {
            _library: library,
            function,
            pipeline,
        })
    }

    /// Returns a fresh mutable encoder. Binding an argument buffer mutates this
    /// object, so it must never be shared by concurrent command recorders.
    fn argument_encoder(&self) -> Retained<ProtocolObject<dyn objc2_metal::MTLArgumentEncoder>> {
        // SAFETY: buffer zero is the kernel's declared argument-buffer
        // parameter, validated while constructing this cache.
        unsafe { self.function.newArgumentEncoderWithBufferIndex(0) }
    }
}

/// Probe the exact runtime compiler, ICB allocation and encoder selectors.
/// No command is submitted, and every temporary native object is discarded.
pub(in crate::backend) fn probe(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    converter: &ConverterCache,
) -> bool {
    use objc2::runtime::NSObjectProtocol as _;
    use objc2_metal::MTLCommandQueue as _;
    if !device.respondsToSelector(objc2::sel!(supportsFamily:)) {
        return false;
    }
    let supported_family = device.supportsFamily(objc2_metal::MTLGPUFamily::Apple3)
        || device.supportsFamily(objc2_metal::MTLGPUFamily::Mac2);
    if !supported_family
        || !device.respondsToSelector(
            objc2::sel!(newIndirectCommandBufferWithDescriptor:maxCommandCount:options:),
        )
        || !device.respondsToSelector(objc2::sel!(newLibraryWithSource:options:error:))
    {
        return false;
    }
    let arguments = converter.argument_encoder();
    if converter.pipeline.threadExecutionWidth() == 0
        || arguments.encodedLength() == 0
        || !arguments.respondsToSelector(objc2::sel!(setIndirectCommandBuffer:atIndex:))
    {
        return false;
    }
    let descriptor = MTLIndirectCommandBufferDescriptor::new();
    descriptor.setInheritBuffers(true);
    descriptor.setInheritPipelineState(true);
    for ty in [
        MTLIndirectCommandType::Draw,
        MTLIndirectCommandType::DrawIndexed,
    ] {
        descriptor.setCommandTypes(ty);
        // SAFETY: a single command is within every supported family's capacity.
        if unsafe {
            device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &descriptor,
                1,
                MTLResourceOptions::StorageModePrivate,
            )
        }
        .is_none()
        {
            return false;
        }
    }
    let pipeline_desc = objc2_metal::MTLRenderPipelineDescriptor::new();
    if !pipeline_desc.respondsToSelector(objc2::sel!(setSupportIndirectCommandBuffers:)) {
        return false;
    }
    let Some(queue) = device.newCommandQueue() else {
        return false;
    };
    let Some(command) = queue.commandBuffer() else {
        return false;
    };
    let Some(blit) = command.blitCommandEncoder() else {
        return false;
    };
    let blit_ok = blit.respondsToSelector(objc2::sel!(resetCommandsInBuffer:withRange:));
    blit.endEncoding();
    let Some(compute) = command.computeCommandEncoder() else {
        return false;
    };
    let compute_ok = compute.respondsToSelector(objc2::sel!(useResource:usage:))
        && compute.respondsToSelector(objc2::sel!(dispatchThreadgroups:threadsPerThreadgroup:));
    compute.endEncoding();
    // SAFETY: the one-pixel descriptor has valid dimensions and format.
    let texture_desc = unsafe {
        objc2_metal::MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            objc2_metal::MTLPixelFormat::RGBA8Unorm,
            1,
            1,
            false,
        )
    };
    texture_desc.setUsage(objc2_metal::MTLTextureUsage::RenderTarget);
    let Some(texture) = device.newTextureWithDescriptor(&texture_desc) else {
        return false;
    };
    let render_desc = objc2_metal::MTLRenderPassDescriptor::renderPassDescriptor();
    // SAFETY: color attachment zero exists in every Metal render descriptor.
    let attachment = unsafe { render_desc.colorAttachments().objectAtIndexedSubscript(0) };
    attachment.setTexture(Some(&texture));
    let Some(render) = command.renderCommandEncoderWithDescriptor(&render_desc) else {
        return false;
    };
    let render_ok = render.respondsToSelector(objc2::sel!(executeCommandsInBuffer:withRange:))
        && render.respondsToSelector(objc2::sel!(useResource:usage:stages:));
    render.endEncoding();
    blit_ok && compute_ok && render_ok
}

#[repr(C)]
struct Parameters {
    argument_word_offset: u64,
    count_word_offset: u64,
    index_byte_offset: u64,
    index_elements: u64,
    stride_words: u32,
    first_command: u32,
    maximum: u32,
    chunk_size: u32,
    index_size: u32,
}

/// Retains resources indirectly referenced by the generated commands.
#[derive(Clone)]
pub(in crate::backend) struct PreparedDraws {
    pub chunks: Vec<(Icb, u32)>,
    _argument_buffers: Vec<Buffer>,
    _args: Buffer,
    _count: Buffer,
    pub index: Option<Buffer>,
}

impl PreparedDraws {
    /// Encode the ICBs at the original draw position in the active pass.
    /// The caller must retain this value until the render command completes.
    pub fn encode_render(&self, render: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>) {
        use objc2_metal::MTLRenderCommandEncoder as _;
        if let Some(index) = &self.index {
            let resource = ProtocolObject::<dyn MTLResource>::from_ref(&**index);
            render.useResource_usage_stages(
                resource,
                MTLResourceUsage::Read,
                objc2_metal::MTLRenderStages::Vertex,
            );
        }
        for (icb, len) in &self.chunks {
            // SAFETY: the range covers exactly the ICB capacity; preparation
            // reset every slot and fills at most one command per slot.
            unsafe {
                render.executeCommandsInBuffer_withRange(icb, NSRange::new(0, *len as usize))
            };
        }
    }
}

pub(in crate::backend) fn prepare(
    converter: &ConverterCache,
    command: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    args: &MetalBuffer,
    args_offset: u64,
    count: &MetalBuffer,
    count_offset: u64,
    max: u32,
    stride: u32,
    index: Option<(&Buffer, u64, u64, IndexFormat)>,
) -> api::Result<PreparedDraws> {
    super::validate(
        args.usage,
        args.size(),
        args_offset,
        count.usage,
        count.size(),
        count_offset,
        max,
        stride,
        index.is_some(),
    )?;
    let device = command.device();
    let argument_encoder = converter.argument_encoder();
    let mut result = PreparedDraws {
        chunks: Vec::new(),
        _argument_buffers: Vec::new(),
        _args: args.raw.clone(),
        _count: count.raw.clone(),
        index: index.as_ref().map(|(buffer, _, _, _)| (*buffer).clone()),
    };
    let (index_raw, index_offset, index_elements, index_size) = match index {
        Some((buffer, offset, size, format)) => {
            let width = match format {
                IndexFormat::Uint16 => 2,
                IndexFormat::Uint32 => 4,
            };
            if offset % width != 0
                || offset
                    .checked_add(size)
                    .is_none_or(|end| end > buffer.length() as u64)
            {
                return Err(api::Error::InvalidRange);
            }
            (buffer, offset, size / width, width as u32)
        }
        None => (&args.raw, 0, 0, 0),
    };
    for (first, len) in super::chunks(max) {
        let descriptor = MTLIndirectCommandBufferDescriptor::new();
        descriptor.setCommandTypes(if index_size == 0 {
            MTLIndirectCommandType::Draw
        } else {
            MTLIndirectCommandType::DrawIndexed
        });
        descriptor.setInheritBuffers(true);
        descriptor.setInheritPipelineState(true);
        // SAFETY: len is nonzero and no greater than the Metal 3 ICB limit.
        let icb = unsafe {
            device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &descriptor,
                len as usize,
                MTLResourceOptions::StorageModePrivate,
            )
        }
        .ok_or(api::Error::OutOfMemory)?;
        let arg_buffer = device
            .newBufferWithLength_options(
                argument_encoder.encodedLength(),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: the argument buffer was allocated to the encoder's exact size,
        // and slot zero is the ICB declared by the kernel.
        unsafe {
            argument_encoder.setArgumentBuffer_offset(Some(&arg_buffer), 0);
            argument_encoder.setIndirectCommandBuffer_atIndex(Some(&icb), 0);
        }
        let blit = command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: the whole range is within the newly allocated ICB.
        unsafe { blit.resetCommandsInBuffer_withRange(&icb, NSRange::new(0, len as usize)) };
        blit.endEncoding();
        let compute = command
            .computeCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        compute.setComputePipelineState(&converter.pipeline);
        let params = Parameters {
            argument_word_offset: args_offset / 4,
            count_word_offset: count_offset / 4,
            index_byte_offset: index_offset,
            index_elements,
            stride_words: stride / 4,
            first_command: first,
            maximum: max,
            chunk_size: len,
            index_size,
        };
        // SAFETY: all bindings use offset zero and the buffers are retained by
        // PreparedDraws; the kernel checks its index against the validated extents.
        unsafe {
            compute.setBuffer_offset_atIndex(Some(&arg_buffer), 0, 0);
            compute.setBuffer_offset_atIndex(Some(&args.raw), 0, 1);
            compute.setBuffer_offset_atIndex(Some(&count.raw), 0, 2);
            compute.setBuffer_offset_atIndex(Some(index_raw), 0, 3);
            compute.setBytes_length_atIndex(
                NonNull::from(&params).cast(),
                std::mem::size_of::<Parameters>(),
                4,
            );
        }
        let resource = ProtocolObject::<dyn MTLResource>::from_ref(&*icb);
        compute.useResource_usage(resource, MTLResourceUsage::Write);
        let width = converter.pipeline.threadExecutionWidth().max(1);
        compute.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (len as usize).div_ceil(width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        compute.endEncoding();
        result.chunks.push((icb, len));
        result._argument_buffers.push(arg_buffer);
    }
    Ok(result)
}
