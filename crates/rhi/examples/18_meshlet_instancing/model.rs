//! Read the Microsoft MSHL model container used by the meshlet sample.

#[derive(Clone, Copy)]
struct Accessor {
    view: usize,
    size: u32,
    count: u32,
}

pub struct Mesh<'a> {
    pub vertices: &'a [u8],
    pub meshlets: &'a [u8],
    pub unique_indices: &'a [u8],
    pub primitives: &'a [u8],
    pub index_size: u32,
    pub meshlet_count: u32,
}

fn word(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?,
    ))
}

/// Parse the single-mesh, single-subset model used by the reference sample.
pub fn load(bytes: &[u8]) -> Option<Mesh<'_>> {
    if word(bytes, 0)? != 0x4d53_484c || word(bytes, 4)? != 0 || word(bytes, 8)? != 1 {
        return None;
    }
    let accessor_count = word(bytes, 12)? as usize;
    let view_count = word(bytes, 16)? as usize;
    let data_size = word(bytes, 20)? as usize;
    let mesh = 24usize;
    let accessors = mesh.checked_add(48)?;
    let views = accessors.checked_add(accessor_count.checked_mul(20)?)?;
    let data_offset = views.checked_add(view_count.checked_mul(8)?)?;
    let data = bytes.get(data_offset..data_offset.checked_add(data_size)?)?;
    if data_offset + data_size != bytes.len() {
        return None;
    }
    let accessor = |index: u32| -> Option<Accessor> {
        let index = index as usize;
        if index >= accessor_count {
            return None;
        }
        let offset = accessors.checked_add(index.checked_mul(20)?)?;
        Some(Accessor {
            view: word(bytes, offset)? as usize,
            size: word(bytes, offset + 8)?,
            count: word(bytes, offset + 16)?,
        })
    };
    let get = |a: Accessor| -> Option<&[u8]> {
        if a.view >= view_count {
            return None;
        }
        let offset = views.checked_add(a.view.checked_mul(8)?)?;
        let start = word(bytes, offset)? as usize;
        let size = word(bytes, offset + 4)? as usize;
        data.get(start..start.checked_add(size)?)
    };
    let indices = accessor(word(bytes, mesh)?)?;
    let vertices = accessor(word(bytes, mesh + 8)?)?;
    let meshlets = accessor(word(bytes, mesh + 28)?)?;
    let subset = get(accessor(word(bytes, mesh + 32)?)?)?;
    if word(subset, 0)? != 0 || word(subset, 4)? != meshlets.count {
        return None;
    }
    Some(Mesh {
        vertices: get(vertices)?,
        meshlets: get(meshlets)?,
        unique_indices: get(accessor(word(bytes, mesh + 36)?)?)?,
        primitives: get(accessor(word(bytes, mesh + 40)?)?)?,
        index_size: indices.size,
        meshlet_count: meshlets.count,
    })
}
