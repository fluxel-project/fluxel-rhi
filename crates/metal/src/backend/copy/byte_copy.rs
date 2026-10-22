//! Byte-granular GPU copies for Metal blits with stricter offset alignment.

use fluxel_rhi_core::api;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder as _, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder as _,
    MTLComputeCommandEncoder as _, MTLDevice as _, MTLLibrary as _, MTLSize,
};
use std::ptr::NonNull;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Params { ulong source; ulong destination; ulong width; ulong rows; ulong source_stride; ulong destination_stride; ulong word_base; ulong word_count; };
kernel void copy_bytes(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                       constant Params& p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (ulong(id) >= p.word_count) return;
    ulong word = p.word_base + ulong(id) * 4;
    for (ulong i = 0; i < 4; ++i) {
        ulong address = word + i;
        if (address < p.destination) continue;
        ulong relative = address - p.destination;
        ulong row = relative / p.destination_stride;
        ulong column = relative % p.destination_stride;
        if (row < p.rows && column < p.width)
            dst[address] = src[p.source + row * p.source_stride + column];
    }
}
"#;

#[repr(C)]
struct Params {
    source: u64,
    destination: u64,
    width: u64,
    rows: u64,
    source_stride: u64,
    destination_stride: u64,
    word_base: u64,
    word_count: u64,
}

/// Source and destination must be in range and must not overlap. A single GPU
/// thread owns a whole aligned destination word, including partial edge words.
pub(in crate::backend) fn copy_bytes(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    src: &ProtocolObject<dyn MTLBuffer>,
    source: usize,
    dst: &ProtocolObject<dyn MTLBuffer>,
    destination: usize,
    bytes: usize,
) -> api::Result<()> {
    copy_rows(
        command,
        src,
        source,
        bytes,
        dst,
        destination,
        bytes,
        bytes,
        1,
    )
}

pub(super) fn copy_rows(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    src: &ProtocolObject<dyn MTLBuffer>,
    source: usize,
    source_stride: usize,
    dst: &ProtocolObject<dyn MTLBuffer>,
    destination: usize,
    destination_stride: usize,
    width: usize,
    rows: usize,
) -> api::Result<()> {
    if rows == 0 || width == 0 {
        return Ok(());
    }
    if [
        source,
        source_stride,
        destination,
        destination_stride,
        width,
    ]
    .iter()
    .all(|v| v % 4 == 0)
    {
        let blit = command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        for row in 0..rows {
            // SAFETY: the caller validated each row and the aligned offsets.
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src,
                    source + row * source_stride,
                    dst,
                    destination + row * destination_stride,
                    width,
                );
            }
        }
        blit.endEncoding();
        return Ok(());
    }
    let device = command.device();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
        .map_err(|e| api::Error::Shader(format!("Metal byte copy shader: {e:?}")))?;
    let function = library
        .newFunctionWithName(&NSString::from_str("copy_bytes"))
        .ok_or_else(|| api::Error::Shader("Metal byte copy function missing".into()))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| api::Error::Backend(format!("Metal byte copy pipeline: {e:?}")))?;
    let compute = command
        .computeCommandEncoder()
        .ok_or(api::Error::OutOfMemory)?;
    let end = destination
        .checked_add(
            (rows - 1)
                .checked_mul(destination_stride)
                .ok_or(api::Error::InvalidRange)?,
        )
        .and_then(|v| v.checked_add(width))
        .ok_or(api::Error::InvalidRange)?;
    let mut word = destination & !3;
    let mut params = Params {
        source: source as u64,
        destination: destination as u64,
        width: width as u64,
        rows: rows as u64,
        source_stride: source_stride as u64,
        destination_stride: destination_stride as u64,
        word_base: word as u64,
        word_count: 0,
    };
    compute.setComputePipelineState(&pipeline);
    // SAFETY: caller validated both ranges; command buffer retains bound objects.
    unsafe {
        compute.setBuffer_offset_atIndex(Some(src), 0, 0);
        compute.setBuffer_offset_atIndex(Some(dst), 0, 1);
    }
    while word < end {
        let words = (end - word).div_ceil(4).min(1 << 20);
        params.word_base = word as u64;
        params.word_count = words as u64;
        // SAFETY: Metal copies the parameter bytes at encode time.
        unsafe {
            compute.setBytes_length_atIndex(
                NonNull::from(&params).cast(),
                std::mem::size_of::<Params>(),
                2,
            );
        }
        compute.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: words.div_ceil(64),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        word = word
            .checked_add(words * 4)
            .ok_or(api::Error::InvalidRange)?;
    }
    compute.endEncoding();
    Ok(())
}
