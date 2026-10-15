//! Fixed Metal buffer-slot partition shared by shader bindings and vertex input.
//!
//! WGSL bind groups occupy buffer slots 0–27. Vertex data uses the three
//! remaining practical slots 28–30, preventing `setVertexBuffer` from
//! overwriting a uniform buffer bound by a raster bind group. Texture and
//! sampler indices are separate Metal namespaces and retain the bind-group map.

#[cfg(any(target_vendor = "apple", test))]
use fluxel_rhi_core::api;

pub(crate) const MAX_VERTEX_BUFFERS: usize = 3;
pub(crate) const MAX_VERTEX_ATTRIBUTES: u32 = 31;
#[cfg(any(target_vendor = "apple", test))]
const VERTEX_BUFFER_BASE: usize = 28;

/// Maps the public vertex-buffer slot to the disjoint native Metal range.
#[cfg(any(target_vendor = "apple", test))]
pub(crate) fn vertex_buffer_index(slot: u32) -> api::Result<usize> {
    let slot = usize::try_from(slot).map_err(|_| api::Error::InvalidRange)?;
    if slot >= MAX_VERTEX_BUFFERS {
        return Err(api::Error::InvalidRange);
    }
    Ok(VERTEX_BUFFER_BASE + slot)
}

#[cfg(test)]
mod tests {
    use super::{MAX_VERTEX_ATTRIBUTES, MAX_VERTEX_BUFFERS, vertex_buffer_index};
    use fluxel_rhi_core::api::Error;

    #[test]
    fn vertex_slots_do_not_overlap_resource_binding_buffers() {
        assert_eq!(vertex_buffer_index(0), Ok(28));
        assert_eq!(vertex_buffer_index(1), Ok(29));
        assert_eq!(vertex_buffer_index(2), Ok(30));
        assert_eq!(MAX_VERTEX_BUFFERS, 3);
        assert_eq!(MAX_VERTEX_ATTRIBUTES, 31);
        assert!(matches!(vertex_buffer_index(3), Err(Error::InvalidRange)));
    }
}
