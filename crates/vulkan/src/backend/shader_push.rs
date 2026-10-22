//! Relocate the direct members of SPIR-V push-constant blocks for one stage.
use super::*;
use std::collections::{HashMap, HashSet};

pub(super) fn relocate(
    device: &VulkanDevice,
    shader: VulkanShader,
    base: u32,
) -> Result<VulkanShader> {
    if base == 0 {
        return Ok(shader);
    }
    let mut words = shader.0.words.clone();
    let mut pointers = HashMap::new();
    let mut variables = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        if count == 0 || offset + count > words.len() {
            return Err(Error::Shader("invalid SPIR-V instruction".into()));
        }
        let inst = &words[offset..offset + count];
        match inst[0] & 0xffff {
            32 if count >= 4 && inst[2] == 9 => {
                pointers.insert(inst[1], inst[3]);
            }
            59 if count >= 4 && inst[3] == 9 => {
                variables.push(inst[1]);
            }
            _ => {}
        }
        offset += count;
    }
    let structs: HashSet<_> = variables
        .into_iter()
        .map(|v| pointers.get(&v).copied().ok_or(Error::InvalidRange))
        .collect::<Result<_>>()?;
    if structs.is_empty() {
        return Ok(shader);
    }
    // A block type may also be used by a uniform or storage declaration. Clone
    // only the push block and its pointer so its member offsets can move without
    // changing the other declaration's ABI.
    let mut shared = HashSet::new();
    offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        let inst = &words[offset..offset + count];
        match inst[0] & 0xffff {
            32 if count >= 4 && inst[2] != 9 && structs.contains(&inst[3]) => {
                shared.insert(inst[3]);
            }
            30 if count >= 3 => {
                shared.extend(inst[2..].iter().filter(|id| structs.contains(id)).copied());
            }
            28 | 29 if count >= 3 && structs.contains(&inst[2]) => {
                shared.insert(inst[2]);
            }
            _ => {}
        }
        offset += count;
    }
    let structs = if shared.is_empty() {
        structs
    } else {
        words = clone_shared_blocks(&words, &pointers, &shared)?;
        structs
            .into_iter()
            .map(|id| {
                // clone_shared_blocks allocates cloned struct IDs in sorted order.
                let mut sorted: Vec<_> = shared.iter().copied().collect();
                sorted.sort_unstable();
                sorted
                    .iter()
                    .position(|v| *v == id)
                    .map_or(id, |index| shader.0.words[3] + index as u32)
            })
            .collect()
    };
    let mut group_decorations: HashMap<u32, Vec<Vec<u32>>> = HashMap::new();
    offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        let inst = &words[offset..offset + count];
        if inst[0] & 0xffff == 71 && count >= 3 {
            group_decorations
                .entry(inst[1])
                .or_default()
                .push(inst[2..].to_vec());
        }
        offset += count;
    }
    let mut rewritten = words[..5].to_vec();
    offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        let inst = &words[offset..offset + count];
        let op = inst[0] & 0xffff;
        if op == 75 && count >= 2 && inst[2..].chunks_exact(2).any(|v| structs.contains(&v[0])) {
            let decorations = group_decorations
                .get(&inst[1])
                .ok_or_else(|| Error::Shader("missing SPIR-V decoration group".into()))?;
            let mut remaining = Vec::new();
            for pair in inst[2..].chunks_exact(2) {
                if !structs.contains(&pair[0]) {
                    remaining.extend_from_slice(pair);
                    continue;
                }
                for decoration in decorations {
                    let len = 3 + decoration.len();
                    rewritten.push(((len as u32) << 16) | 72);
                    rewritten.extend_from_slice(pair);
                    rewritten.extend_from_slice(decoration);
                    if decoration[0] == 35 {
                        let value = rewritten.last_mut().ok_or(Error::InvalidRange)?;
                        *value = value.checked_add(base).ok_or(Error::InvalidRange)?;
                    }
                }
            }
            if !remaining.is_empty() {
                rewritten.push((((remaining.len() + 2) as u32) << 16) | 75);
                rewritten.push(inst[1]);
                rewritten.extend(remaining);
            }
        } else {
            let start = rewritten.len();
            rewritten.extend_from_slice(inst);
            if op == 72 && count >= 5 && structs.contains(&inst[1]) && inst[3] == 35 {
                rewritten[start + 4] = rewritten[start + 4]
                    .checked_add(base)
                    .ok_or(Error::InvalidRange)?;
            }
        }
        offset += count;
    }
    words = rewritten;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    // SAFETY: words is valid storage through creation and the device is retained.
    let module = unsafe { device.0.device.create_shader_module(&info, None) }.map_err(vk_error)?;
    Ok(VulkanShader(Arc::new(ShaderState {
        device: device.0.clone(),
        module,
        entries: shader.0.entries.clone(),
        words,
        wgsl: None,
    })))
}

/// Clone shared block declarations and preserve the original type for other
/// storage classes. Whole-block loads are reconstructed with the original value
/// type so users of that value keep their existing types.
fn clone_shared_blocks(
    words: &[u32],
    pointers: &HashMap<u32, u32>,
    shared: &HashSet<u32>,
) -> Result<Vec<u32>> {
    let mut sorted: Vec<_> = shared.iter().copied().collect();
    sorted.sort_unstable();
    let mut next = words[3];
    let mut alloc = || {
        let id = next;
        next += 1;
        id
    };
    let clones: HashMap<_, _> = sorted.iter().map(|id| (*id, alloc())).collect();
    let mut pointer_clones = HashMap::new();
    let mut pointer_ids: Vec<_> = pointers
        .iter()
        .filter(|(_, ty)| shared.contains(ty))
        .map(|(id, _)| *id)
        .collect();
    pointer_ids.sort_unstable();
    for id in pointer_ids {
        pointer_clones.insert(id, alloc());
    }
    let mut members = HashMap::new();
    let mut variables = HashMap::new();
    let mut cursor = 5;
    while cursor < words.len() {
        let count = (words[cursor] >> 16) as usize;
        let inst = &words[cursor..cursor + count];
        match inst[0] & 0xffff {
            30 if shared.contains(&inst[1]) => {
                members.insert(inst[1], inst[2..].to_vec());
            }
            59 if count >= 4 && inst[3] == 9 && pointer_clones.contains_key(&inst[1]) => {
                variables.insert(inst[2], pointers[&inst[1]]);
            }
            55 | 57 | 61 | 65 | 66 | 67 | 70 | 83 | 169 | 245
                if count >= 3 && pointer_clones.contains_key(&inst[1]) =>
            {
                variables.insert(inst[2], pointers[&inst[1]]);
            }
            _ => {}
        }
        cursor += count;
    }
    let mut out = words[..5].to_vec();
    cursor = 5;
    while cursor < words.len() {
        let count = (words[cursor] >> 16) as usize;
        let inst = &words[cursor..cursor + count];
        let op = inst[0] & 0xffff;
        let whole_load = op == 61 && count >= 4 && variables.contains_key(&inst[3]);
        let whole_copy = op == 63 && count >= 3 && variables.contains_key(&inst[2]);
        if whole_copy && variables.contains_key(&inst[1]) {
            return Err(Error::Shader(
                "push constants cannot be copy destinations".into(),
            ));
        }
        if whole_load || whole_copy {
            let source = if whole_load { inst[3] } else { inst[2] };
            let original = variables[&source];
            let fields = members.get(&original).ok_or(Error::InvalidRange)?;
            let loaded = alloc();
            let (target_access, source_access) = if whole_copy {
                copy_memory_operands(&inst[3..])?
            } else {
                (Vec::new(), inst[4..].to_vec())
            };
            out.push((((4 + source_access.len()) as u32) << 16) | 61);
            out.extend_from_slice(&[clones[&original], loaded, source]);
            out.extend(source_access);
            let mut values = Vec::with_capacity(fields.len());
            for (index, ty) in fields.iter().enumerate() {
                let value = alloc();
                out.extend_from_slice(&[(5 << 16) | 81, *ty, value, loaded, index as u32]);
                values.push(value);
            }
            let result = if whole_load { inst[2] } else { alloc() };
            out.push((((3 + values.len()) as u32) << 16) | 80);
            out.extend_from_slice(&[original, result]);
            out.extend(values);
            if whole_copy {
                out.push((((3 + target_access.len()) as u32) << 16) | 62);
                out.extend_from_slice(&[inst[1], result]);
                out.extend(target_access);
            }
        } else {
            let mut rewritten = inst.to_vec();
            if op == 59 && count >= 4 && inst[3] == 9 {
                if let Some(id) = pointer_clones.get(&inst[1]) {
                    rewritten[1] = *id;
                }
            }
            if matches!(op, 55 | 57 | 61 | 65 | 66 | 67 | 70 | 83 | 169 | 245) && count >= 3 {
                if let Some(id) = pointer_clones.get(&inst[1]) {
                    rewritten[1] = *id;
                }
            }
            if op == 33 {
                for ty in &mut rewritten[2..] {
                    if let Some(id) = pointer_clones.get(ty) {
                        *ty = *id;
                    }
                }
            }
            out.extend(rewritten);
        }
        // Duplicate annotations before their type declarations, retaining group
        // memberships and every member decoration on both block types.
        match op {
            71 | 72 if count >= 2 && clones.contains_key(&inst[1]) => {
                let mut copy = inst.to_vec();
                copy[1] = clones[&inst[1]];
                out.extend(copy);
            }
            74 if count >= 3 => {
                let targets: Vec<_> = inst[2..]
                    .iter()
                    .filter_map(|id| clones.get(id).copied())
                    .collect();
                if !targets.is_empty() {
                    out.push((((2 + targets.len()) as u32) << 16) | 74);
                    out.push(inst[1]);
                    out.extend(targets);
                }
            }
            75 if count >= 4 => {
                let targets: Vec<_> = inst[2..]
                    .chunks_exact(2)
                    .filter_map(|pair| clones.get(&pair[0]).map(|id| [*id, pair[1]]))
                    .flatten()
                    .collect();
                if !targets.is_empty() {
                    out.push((((2 + targets.len()) as u32) << 16) | 75);
                    out.push(inst[1]);
                    out.extend(targets);
                }
            }
            30 if count >= 2 && clones.contains_key(&inst[1]) => {
                let mut copy = inst.to_vec();
                copy[1] = clones[&inst[1]];
                out.extend(copy);
            }
            32 if count >= 4 && pointer_clones.contains_key(&inst[1]) => {
                let mut copy = inst.to_vec();
                copy[1] = pointer_clones[&inst[1]];
                copy[3] = clones[&inst[3]];
                out.extend(copy);
            }
            _ => {}
        }
        cursor += count;
    }
    out[3] = next;
    Ok(out)
}

/// Split the one shared or two per-pointer SPIR-V memory operand masks.
fn copy_memory_operands(words: &[u32]) -> Result<(Vec<u32>, Vec<u32>)> {
    fn one(words: &[u32]) -> Result<(&[u32], &[u32])> {
        let Some(&mask) = words.first() else {
            return Ok((&[], &[]));
        };
        let extra = (mask & (0x2 | 0x8 | 0x10 | 0x10000 | 0x20000)).count_ones() as usize;
        if words.len() < 1 + extra {
            return Err(Error::Shader("invalid memory operands".into()));
        }
        Ok(words.split_at(1 + extra))
    }
    let (first, rest) = one(words)?;
    if rest.is_empty() {
        fn filtered(words: &[u32], remove: u32) -> Vec<u32> {
            if words.is_empty() {
                return Vec::new();
            }
            let mask = words[0];
            let mut result = vec![mask & !remove];
            let mut index = 1;
            for bit in [0x2, 0x8, 0x10, 0x10000, 0x20000] {
                if mask & bit != 0 {
                    if bit & remove == 0 {
                        result.push(words[index]);
                    }
                    index += 1;
                }
            }
            result
        }
        return Ok((filtered(first, 0x10), filtered(first, 0x8)));
    }
    let (second, rest) = one(rest)?;
    if !rest.is_empty() {
        return Err(Error::Shader("invalid memory operands".into()));
    }
    Ok((first.to_vec(), second.to_vec()))
}
