//! Direct lowering of explicit Core barriers onto one ordered GL context.
//!
//! GL has no texture layouts; these descriptions are consumed without keeping
//! resource history. Incoherent shader access needs native visibility barriers.
//! Khronos GL 4.5 specification section 7.13.2 defines bits by the subsequent
//! access class, not by the earlier shader write.
use crate::backend::{
    api::GlMemoryBarrier as Bits,
    rhi::{Gl, types::GlEncoder},
};
use fluxel_rhi_core::api::{self, Error, ResourceTransition, Result};

const STAGES: u64 = api::StageMask::HOST.0
    | api::StageMask::COPY.0
    | api::StageMask::INDIRECT.0
    | api::StageMask::VERTEX_INPUT.0
    | api::StageMask::VERTEX.0
    | api::StageMask::FRAGMENT.0
    | api::StageMask::COLOR_OUTPUT.0
    | api::StageMask::COMPUTE.0;
const BUFFER: u64 = api::AccessMask::HOST_READ.0
    | api::AccessMask::HOST_WRITE.0
    | api::AccessMask::COPY_READ.0
    | api::AccessMask::COPY_WRITE.0
    | api::AccessMask::VERTEX_READ.0
    | api::AccessMask::INDEX_READ.0
    | api::AccessMask::UNIFORM_READ.0
    | api::AccessMask::SHADER_READ.0
    | api::AccessMask::SHADER_WRITE.0
    | api::AccessMask::INDIRECT_READ.0
    | api::AccessMask::QUERY_RESOLVE_WRITE.0;
const TEXTURE: u64 = api::AccessMask::HOST_READ.0
    | api::AccessMask::HOST_WRITE.0
    | api::AccessMask::COPY_READ.0
    | api::AccessMask::COPY_WRITE.0
    | api::AccessMask::SHADER_READ.0
    | api::AccessMask::SHADER_WRITE.0
    | api::AccessMask::COLOR_READ.0
    | api::AccessMask::COLOR_WRITE.0
    | api::AccessMask::DEPTH_READ.0
    | api::AccessMask::DEPTH_WRITE.0
    | api::AccessMask::STENCIL_READ.0
    | api::AccessMask::STENCIL_WRITE.0;

fn scope(stages: api::StageMask, access: api::AccessMask, allowed: u64) -> Result<u64> {
    let stages = if stages == api::StageMask::ALL_COMMANDS {
        STAGES
    } else {
        stages.0
    };
    let accesses = if access == api::AccessMask::ALL {
        allowed
    } else {
        access.0
    };
    if stages & !STAGES != 0 || accesses & !allowed != 0 {
        return Err(Error::Unsupported("GL transition stage or access"));
    }
    if stages == 0 && accesses != 0 {
        return Err(Error::InvalidState);
    }
    use api::{AccessMask as A, StageMask as S};
    for (mask, possible_stages) in [
        (A::HOST_READ.0 | A::HOST_WRITE.0, S::HOST.0),
        (
            A::COPY_READ.0 | A::COPY_WRITE.0 | A::QUERY_RESOLVE_WRITE.0,
            S::COPY.0,
        ),
        (A::VERTEX_READ.0 | A::INDEX_READ.0, S::VERTEX_INPUT.0),
        (
            A::UNIFORM_READ.0 | A::SHADER_READ.0 | A::SHADER_WRITE.0,
            S::VERTEX.0 | S::FRAGMENT.0 | S::COMPUTE.0,
        ),
        (A::INDIRECT_READ.0, S::INDIRECT.0),
        (A::COLOR_READ.0 | A::COLOR_WRITE.0, S::COLOR_OUTPUT.0),
        (
            A::DEPTH_READ.0 | A::DEPTH_WRITE.0 | A::STENCIL_READ.0 | A::STENCIL_WRITE.0,
            S::FRAGMENT.0 | S::COLOR_OUTPUT.0,
        ),
    ] {
        if accesses & mask != 0 && stages & possible_stages == 0 {
            return Err(Error::InvalidState);
        }
    }
    Ok(accesses)
}
fn destination(access: u64, texture: bool, layout: Option<api::ImageLayout>) -> Bits {
    use api::AccessMask as A;
    let mut bits = Bits(0);
    let mut add = |mask: u64, bit: Bits| {
        if access & mask != 0 {
            bits = bits.union(bit);
        }
    };
    if texture {
        if access & A::SHADER_READ.0 != 0 {
            bits = bits.union(Bits::TEXTURE_FETCH);
            if layout != Some(api::ImageLayout::ShaderReadOnly) {
                bits = bits.union(Bits::SHADER_IMAGE_ACCESS);
            }
        }
        if access & A::SHADER_WRITE.0 != 0 {
            bits = bits.union(Bits::SHADER_IMAGE_ACCESS);
        }
        if access & (A::COPY_READ.0 | A::COPY_WRITE.0 | A::HOST_READ.0 | A::HOST_WRITE.0) != 0 {
            bits = bits.union(Bits::TEXTURE_UPDATE).union(Bits::FRAMEBUFFER);
        }
        if access
            & (A::COLOR_READ.0
                | A::COLOR_WRITE.0
                | A::DEPTH_READ.0
                | A::DEPTH_WRITE.0
                | A::STENCIL_READ.0
                | A::STENCIL_WRITE.0)
            != 0
        {
            bits = bits.union(Bits::FRAMEBUFFER);
        }
    } else {
        add(A::VERTEX_READ.0, Bits::VERTEX_ATTRIB_ARRAY);
        add(A::INDEX_READ.0, Bits::ELEMENT_ARRAY);
        add(A::UNIFORM_READ.0, Bits::UNIFORM);
        add(A::SHADER_READ.0 | A::SHADER_WRITE.0, Bits::SHADER_STORAGE);
        add(A::INDIRECT_READ.0, Bits::COMMAND);
        add(
            A::COPY_READ.0 | A::COPY_WRITE.0,
            Bits::BUFFER_UPDATE.union(Bits::PIXEL_BUFFER),
        );
        add(A::HOST_READ.0 | A::HOST_WRITE.0, Bits::BUFFER_UPDATE);
        add(
            A::QUERY_RESOLVE_WRITE.0,
            Bits::QUERY_BUFFER.union(Bits::BUFFER_UPDATE),
        );
    }
    bits
}

pub(super) fn transition(
    encoder: &mut GlEncoder,
    ops: &[ResourceTransition<'_, Gl>],
) -> Result<()> {
    let mut bits = Bits(0);
    let mut needs_buffer = false;
    let mut needs_image = false;
    let mut global_shader = false;
    for op in ops {
        let (before, after, texture, layout) = match op {
            ResourceTransition::Buffer(b) => {
                // SAFETY: Core keeps the target borrowed for this call.
                let resource = unsafe { b.target.buffer.native_ref() };
                if !std::rc::Rc::ptr_eq(&encoder.state, &resource.state) {
                    return Err(Error::WrongDevice);
                }
                if b.target.range.start >= b.target.range.end
                    || b.target.range.end > resource.desc.size
                {
                    return Err(Error::InvalidRange);
                }
                if b.ownership.is_some() {
                    return Err(Error::Unsupported("GL queue ownership transfer"));
                }
                (
                    scope(b.usage.from.stages, b.usage.from.accesses, BUFFER)?,
                    scope(b.usage.to.stages, b.usage.to.accesses, BUFFER)?,
                    false,
                    None,
                )
            }
            ResourceTransition::Texture(b) => {
                // SAFETY: Core keeps the target borrowed for this call.
                let resource = unsafe { b.target.texture.native_ref() };
                if !std::rc::Rc::ptr_eq(&encoder.state, &resource.state) {
                    return Err(Error::WrongDevice);
                }
                let layers =
                    if resource.desc.dimension == crate::backend::api::GlTextureDimension::D3 {
                        1
                    } else {
                        resource.desc.extent.depth_or_layers
                    };
                use crate::backend::api::GlFormat as F;
                use api::TextureAspects as A;
                let aspects = match resource.desc.format {
                    F::Depth16Unorm | F::Depth24Unorm | F::Depth32Float => A::DEPTH,
                    F::Depth24PlusStencil8 | F::Depth32FloatStencil8 => A::DEPTH | A::STENCIL,
                    F::Stencil8 => A::STENCIL,
                    _ => A::COLOR,
                };
                if b.target.mips.start >= b.target.mips.end
                    || b.target.mips.end > resource.desc.mip_level_count
                    || b.target.layers.start >= b.target.layers.end
                    || b.target.layers.end > layers
                    || b.target.aspects.0 == 0
                    || b.target.aspects.0 & !aspects.0 != 0
                {
                    return Err(Error::InvalidRange);
                }
                if b.ownership.is_some() {
                    return Err(Error::Unsupported("GL queue ownership transfer"));
                }
                let from = scope(b.usage.from.stages, b.usage.from.accesses, TEXTURE)?;
                let to = scope(b.usage.to.stages, b.usage.to.accesses, TEXTURE)?;
                if b.layout.is_some_and(|(a, b)| {
                    (a == api::ImageLayout::Undefined && from != 0)
                        || b == api::ImageLayout::Undefined
                }) {
                    return Err(Error::InvalidState);
                }
                (from, to, true, b.layout.map(|(_, after)| after))
            }
            ResourceTransition::Global { before, after } => {
                let from = scope(before.stages, before.accesses, BUFFER | TEXTURE)?;
                let to = scope(after.stages, after.accesses, BUFFER | TEXTURE)?;
                if (from | to) & api::AccessMask::SHADER_WRITE.0 != 0 {
                    bits = bits
                        .union(destination(to & BUFFER, false, None))
                        .union(destination(to & TEXTURE, true, None));
                    global_shader = true;
                }
                continue;
            }
            _ => return Err(Error::Unsupported("GL resource transition kind")),
        };
        if (before | after) & api::AccessMask::SHADER_WRITE.0 != 0 {
            bits = bits.union(destination(after, texture, layout));
            if texture {
                needs_image = true;
            } else {
                needs_buffer = true;
            }
        }
    }
    if bits.0 == 0 {
        return Ok(());
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::api::{
            GlCapability, GlFamilyApi as _, GlFamilyProfile, GlSyncApi as _,
        };
        let result = encoder
            .state
            .with_native("explicit GL memory barrier", |provider| {
                let caps = provider.discovery().capabilities();
                let has_buffer = caps.supports(GlCapability::StorageBuffer);
                let has_image = caps.supports(GlCapability::StorageImage);
                if (global_shader && !has_buffer && !has_image)
                    || (needs_buffer && !has_buffer)
                    || (needs_image && !has_image)
                {
                    return Err(crate::backend::api::GlError::Unsupported {
                        operation: "explicit GL memory barrier",
                        reason: "shader storage capability",
                    });
                }
                let mut usable = bits;
                // QUERY_BUFFER_BARRIER_BIT was introduced in GL 4.4 / ARB_query_buffer_object;
                // the portable query resolve path uses a buffer update instead.
                let query_buffer = matches!(provider.discovery().context().profile(), GlFamilyProfile::Desktop { major, minor } if major > 4 || (major == 4 && minor >= 4))
                    || (matches!(provider.discovery().context().profile(), GlFamilyProfile::Desktop { .. })
                        && provider.discovery().extensions().is_acquired(
                            crate::backend::api::GlKnownExtension::ArbQueryBufferObject,
                        ));
                if !query_buffer { usable.0 &= !Bits::QUERY_BUFFER.0; }
                if global_shader && !has_buffer {
                    usable.0 &= !Bits::SHADER_STORAGE.0;
                }
                if global_shader && !has_image {
                    usable.0 &= !Bits::SHADER_IMAGE_ACCESS.0;
                }
                provider.memory_barrier(usable)
            })
            .map_err(super::super::spi::native_error);
        if result.is_ok() {
            encoder.issued = true;
        } else {
            encoder.indeterminate = true;
        }
        return result;
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = (encoder, needs_buffer, needs_image, global_shader);
        Err(Error::Unsupported("GL shader memory barrier"))
    }
}

#[cfg(test)]
#[path = "../../../tests/backend/rhi/command/transition__tests.rs"]
mod tests;
