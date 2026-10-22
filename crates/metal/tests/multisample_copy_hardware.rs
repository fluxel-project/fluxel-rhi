//! Real Apple GPU regression for per-sample combined depth/stencil copies.
#![cfg(all(target_vendor = "apple", feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_metal::{MetalExternalHandle, MetalInstance};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCompareFunction,
    MTLComputeCommandEncoder as _, MTLDepthStencilDescriptor, MTLDevice as _, MTLLibrary as _,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLResourceOptions, MTLSize,
    MTLStencilDescriptor, MTLStencilOperation, MTLStorageMode, MTLStoreAction, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};
use std::{
    future::Future,
    ptr::NonNull,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

struct WakeThread(std::thread::Thread);
impl Wake for WakeThread {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<F: Future>(f: F) -> F::Output {
    let w = Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut cx = Context::from_waker(&w);
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park(),
        }
    }
}

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct V { float4 position [[position]]; };
struct D { float depth [[depth(any)]]; };
vertex V vs(uint id [[vertex_id]]) {
  float2 p = float2((id << 1) & 2, id & 2);
  return { float4(p * float2(2,-2) + float2(-1,1),0,1) };
}
fragment D fs(uint s [[sample_id]], constant uint& selected [[buffer(0)]]) {
  if (s != selected) discard_fragment();
  return { float(s + 1) * 0.125 };
}
kernel void read_planes(depth2d_ms<float, access::read> depth [[texture(0)]],
                        texture2d_ms<uint, access::read> stencil [[texture(1)]],
                        device uint2* result [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
  for (uint s = 0; s < 4; ++s) {
    result[(p.y * 4 + p.x) * 4 + s] = uint2(as_type<uint>(depth.read(p,s)), stencil.read(p,s).r);
  }
}
"#;

#[test]
fn combined_msaa_plane_copy_preserves_other_plane_and_outside_region() {
    let device = block_on(MetalInstance::new().unwrap().open_default()).unwrap();
    // SAFETY: the guard retains the native device for the whole test.
    let native_guard = unsafe {
        device
            .native_interop()
            .unwrap()
            .acquire(NativeInteropRequest::Device)
    }
    .unwrap();
    // SAFETY: Metal's native interop returns an MTLDevice object pointer.
    let native = unsafe {
        &*(native_guard.raw_pointer() as *const ProtocolObject<dyn objc2_metal::MTLDevice>)
    };
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Depth32FloatStencil8,
        mip_levels: 1,
        samples: 4,
        usage: TextureUsage::COPY_SRC
            .union(TextureUsage::COPY_DST)
            .union(TextureUsage::RENDER_ATTACHMENT)
            .union(TextureUsage::TEXTURE_BINDING),
        view_formats: vec![],
    };
    if !matches!(device.texture_support(&desc), Support::Supported(_)) {
        return;
    }
    let td = MTLTextureDescriptor::new();
    td.setTextureType(MTLTextureType::Type2DMultisample);
    td.setPixelFormat(MTLPixelFormat::Depth32Float_Stencil8);
    // SAFETY: these fixed dimensions and sample count form a valid MSAA descriptor.
    unsafe {
        td.setWidth(4);
        td.setHeight(4);
        td.setSampleCount(4);
    }
    td.setStorageMode(MTLStorageMode::Private);
    td.setUsage(
        MTLTextureUsage::RenderTarget
            | MTLTextureUsage::ShaderRead
            | MTLTextureUsage::PixelFormatView,
    );
    let source = native.newTextureWithDescriptor(&td).unwrap();
    let dest = native.newTextureWithDescriptor(&td).unwrap();
    let external = device
        .external()
        .expect("Metal external resource capability");
    // SAFETY: both native textures belong to this device and match desc exactly;
    // all accesses are recorded in the same RHI command buffer below.
    let src_import =
        unsafe { external.import_texture(MetalExternalHandle::Texture(source.clone()), &desc) }
            .unwrap();
    let dst_import =
        unsafe { external.import_texture(MetalExternalHandle::Texture(dest.clone()), &desc) }
            .unwrap();
    let src = external.texture(&src_import).unwrap();
    let dst = external.texture(&dst_import).unwrap();
    let library = native
        .newLibraryWithSource_options_error(&NSString::from_str(MSL), None)
        .unwrap();
    let pipeline_desc = MTLRenderPipelineDescriptor::new();
    pipeline_desc.setVertexFunction(
        library
            .newFunctionWithName(&NSString::from_str("vs"))
            .as_deref(),
    );
    pipeline_desc.setFragmentFunction(
        library
            .newFunctionWithName(&NSString::from_str("fs"))
            .as_deref(),
    );
    pipeline_desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float_Stencil8);
    pipeline_desc.setStencilAttachmentPixelFormat(MTLPixelFormat::Depth32Float_Stencil8);
    pipeline_desc.setRasterSampleCount(4);
    let render_pipeline = native
        .newRenderPipelineStateWithDescriptor_error(&pipeline_desc)
        .unwrap();
    let ds = MTLDepthStencilDescriptor::new();
    ds.setDepthCompareFunction(MTLCompareFunction::Always);
    ds.setDepthWriteEnabled(true);
    let stencil = MTLStencilDescriptor::new();
    stencil.setStencilCompareFunction(MTLCompareFunction::Always);
    stencil.setDepthStencilPassOperation(MTLStencilOperation::Replace);
    ds.setFrontFaceStencil(Some(&stencil));
    ds.setBackFaceStencil(Some(&stencil));
    let state = native.newDepthStencilStateWithDescriptor(&ds).unwrap();
    let compute_fn = library
        .newFunctionWithName(&NSString::from_str("read_planes"))
        .unwrap();
    let compute_pipeline = native
        .newComputePipelineStateWithFunction_error(&compute_fn)
        .unwrap();
    let stencil_view = dest
        .newTextureViewWithPixelFormat(MTLPixelFormat::X32_Stencil8)
        .unwrap();
    let output = native
        .newBufferWithLength_options(4 * 4 * 4 * 8, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let lane = device.lanes()[0].id;
    for aspect in [TextureAspects::DEPTH, TextureAspects::STENCIL] {
        let mut encoder = device.encoder(lane).unwrap();
        {
            // SAFETY: the guard exclusively borrows the encoder, native objects stay alive through completion,
            // and all native encoders end before returning control to RHI.
            let guard = unsafe { encoder.native_interop() }.unwrap();
            let command = unsafe {
                &*(guard.raw_pointer() as *const ProtocolObject<dyn objc2_metal::MTLCommandBuffer>)
            };
            for (texture, fill) in [(&source, true), (&dest, false)] {
                let pass = MTLRenderPassDescriptor::renderPassDescriptor();
                let d = pass.depthAttachment();
                d.setTexture(Some(texture));
                d.setLoadAction(MTLLoadAction::Clear);
                d.setClearDepth(0.75);
                d.setStoreAction(MTLStoreAction::Store);
                let s = pass.stencilAttachment();
                s.setTexture(Some(texture));
                s.setLoadAction(MTLLoadAction::Clear);
                s.setClearStencil(42);
                s.setStoreAction(MTLStoreAction::Store);
                let r = command.renderCommandEncoderWithDescriptor(&pass).unwrap();
                if fill {
                    r.setRenderPipelineState(&render_pipeline);
                    r.setDepthStencilState(Some(&state));
                    for sample in 0u32..4 {
                        // SAFETY: the fragment buffer points to this draw's four-byte value.
                        unsafe {
                            r.setFragmentBytes_length_atIndex(NonNull::from(&sample).cast(), 4, 0);
                            r.setStencilReferenceValue(10 + sample);
                            r.drawPrimitives_vertexStart_vertexCount(
                                MTLPrimitiveType::Triangle,
                                0,
                                3,
                            );
                        }
                    }
                }
                r.endEncoding();
            }
        }
        let region = |texture| TextureRegion {
            texture,
            mip: 0,
            origin: [1, 1, 0],
            extent: [2, 2, 1],
            aspect,
        };
        encoder.copy_texture(region(&src), region(&dst)).unwrap();
        {
            // SAFETY: native read follows the copy on the same command buffer; output and textures remain live.
            let guard = unsafe { encoder.native_interop() }.unwrap();
            let command = unsafe {
                &*(guard.raw_pointer() as *const ProtocolObject<dyn objc2_metal::MTLCommandBuffer>)
            };
            let c = command.computeCommandEncoder().unwrap();
            c.setComputePipelineState(&compute_pipeline);
            // SAFETY: buffer and texture indices match the shader, and the dispatch covers its bounded output.
            unsafe {
                c.setTexture_atIndex(Some(&dest), 0);
                c.setTexture_atIndex(Some(&stencil_view), 1);
                c.setBuffer_offset_atIndex(Some(&output), 0, 0);
                c.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: 4,
                        height: 4,
                        depth: 1,
                    },
                    MTLSize {
                        width: 4,
                        height: 4,
                        depth: 1,
                    },
                );
            }
            c.endEncoding();
        }
        let work = encoder.finish().unwrap();
        // SAFETY: work is bound to its recording device and lane, and this test waits for completion.
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: device.id(),
                    ordinal: 1,
                },
                Arc::from([StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                }]),
            )
        };
        let step = plan.step(0);
        let receipt = match unsafe {
            device.queue().submit(SubmitPacket {
                plan,
                steps: vec![BoundStep {
                    step,
                    payload: StepPayload::Commands(vec![work]),
                }],
                completion: CompletionRequest::End,
            })
        } {
            SubmitOutcome::Accepted(r) => r,
            SubmitOutcome::Rejected { error, .. }
            | SubmitOutcome::PartiallyAccepted { error, .. }
            | SubmitOutcome::Indeterminate { error, .. } => panic!("submit failed: {error:?}"),
        };
        block_on(device.wait_completion(&receipt.checkpoints[0].point)).unwrap();
        // SAFETY: the completion covers the shared buffer write and the allocation has 128 u32 values.
        let values =
            unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<u32>(), 128) };
        for y in 0..4 {
            for x in 0..4 {
                for sample in 0..4 {
                    let i = ((y * 4 + x) * 4 + sample) * 2;
                    let inside = (1..3).contains(&x) && (1..3).contains(&y);
                    let depth = if inside && aspect == TextureAspects::DEPTH {
                        (sample + 1) as f32 * 0.125
                    } else {
                        0.75
                    };
                    let stencil = if inside && aspect == TextureAspects::STENCIL {
                        10 + sample as u32
                    } else {
                        42
                    };
                    assert_eq!(
                        f32::from_bits(values[i]),
                        depth,
                        "depth ({x},{y}) sample {sample}"
                    );
                    assert_eq!(values[i + 1], stencil, "stencil ({x},{y}) sample {sample}");
                }
            }
        }
    }
}
