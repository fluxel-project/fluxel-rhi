//! Private sampler descriptor remapping for integer border colours.
use super::*;
use std::collections::{HashMap, HashSet};

pub(super) fn remap(
    device: &VulkanDevice,
    shader: VulkanShader,
    layout: &super::super::pipeline::VulkanPipelineLayout,
) -> Result<VulkanShader> {
    let maps = layout.descriptor_binding_maps();
    if maps
        .iter()
        .all(|(native, mirrors)| mirrors.is_empty() && native.iter().all(|(a, b)| a == b))
    {
        return Ok(shader);
    }
    let flattened = flatten_binding_groups(&shader.0.words)?;
    let words = &flattened;
    let mut instructions = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 || at + count > words.len() {
            return Err(Error::Shader("invalid SPIR-V instruction".into()));
        }
        instructions.push(&words[at..at + count]);
        at += count;
    }
    let mut binding = HashMap::new();
    let mut sets = HashMap::new();
    let mut integer_types = HashSet::new();
    let mut images = HashMap::new();
    let mut sampled = HashMap::new();
    let mut sampler_types = HashSet::new();
    let mut function_types = HashMap::new();
    let mut functions = HashMap::new();
    for inst in &instructions {
        match inst[0] & 0xffff {
            21 if inst.len() >= 2 => {
                integer_types.insert(inst[1]);
            }
            25 if inst.len() >= 3 => {
                images.insert(inst[1], inst[2]);
            }
            27 if inst.len() >= 3 => {
                sampled.insert(inst[1], inst[2]);
            }
            26 if inst.len() >= 2 => {
                sampler_types.insert(inst[1]);
            }
            71 if inst.len() >= 4 && inst[2] == 33 => {
                binding.insert(inst[1], inst[3]);
            }
            71 if inst.len() >= 4 && inst[2] == 34 => {
                sets.insert(inst[1], inst[3]);
            }
            _ => {}
        }
    }
    // Type declarations precede their users, so sampler pointer and array
    // types can be recognized in one pass.
    for inst in &instructions {
        match inst[0] & 0xffff {
            28 | 29 if inst.len() >= 3 && sampler_types.contains(&inst[2]) => {
                sampler_types.insert(inst[1]);
            }
            32 if inst.len() >= 4 && sampler_types.contains(&inst[3]) => {
                sampler_types.insert(inst[1]);
            }
            33 if inst.len() >= 3 => {
                function_types.insert(inst[1], inst[3..].to_vec());
            }
            54 if inst.len() >= 5 => {
                functions.insert(inst[2], inst[4]);
            }
            _ => {}
        }
    }
    let mut vars = HashMap::new();
    for (&id, &public) in &binding {
        let set = sets.get(&id).copied().unwrap_or(0) as usize;
        if let Some((_, mirrors)) = maps.get(set) {
            if let Some(&mirror) = mirrors.get(&public) {
                vars.insert(id, mirror);
            }
        }
    }
    let mut next = words[3];
    let mut clones = HashMap::new();
    for inst in &instructions {
        if inst[0] & 0xffff == 59 && inst.len() >= 4 && vars.contains_key(&inst[2]) {
            clones.insert(inst[2], next);
            next = next.checked_add(1).ok_or(Error::InvalidRange)?;
        }
        if inst[0] & 0xffff == 55 && inst.len() >= 3 && sampler_types.contains(&inst[1]) {
            clones.insert(inst[2], next);
            next = next.checked_add(1).ok_or(Error::InvalidRange)?;
        }
    }
    // Sampler values are opaque; duplicate only descriptor pointer/load chains.
    // Preserve original definitions for float uses of the same sampler.
    let mut changed = true;
    while changed {
        changed = false;
        for inst in &instructions {
            let op = inst[0] & 0xffff;
            if inst.len() >= 4
                && matches!(op, 61 | 65 | 66 | 67 | 83)
                && clones.contains_key(&inst[3])
                && !clones.contains_key(&inst[2])
            {
                clones.insert(inst[2], next);
                next = next.checked_add(1).ok_or(Error::InvalidRange)?;
                changed = true;
            }
            let merge = match op {
                169 if inst.len() >= 6 => {
                    clones.contains_key(&inst[4]) || clones.contains_key(&inst[5])
                }
                245 if inst.len() >= 5 => inst[3..]
                    .chunks_exact(2)
                    .any(|pair| clones.contains_key(&pair[0])),
                _ => false,
            };
            if merge && !clones.contains_key(&inst[2]) {
                clones.insert(inst[2], next);
                next = next.checked_add(1).ok_or(Error::InvalidRange)?;
                changed = true;
            }
        }
    }
    let mut out = words[..5].to_vec();
    out[3] = next;
    for inst in instructions {
        let op = inst[0] & 0xffff;
        let mut primary = inst.to_vec();
        if op == 74 && inst.len() >= 3 {
            primary.extend(inst[2..].iter().filter_map(|id| clones.get(id).copied()));
            primary[0] = ((primary.len() as u32) << 16) | op;
        }
        if op == 33 && inst.len() >= 3 {
            primary.truncate(3);
            for ty in &inst[3..] {
                primary.push(*ty);
                if sampler_types.contains(ty) {
                    primary.push(*ty);
                }
            }
            primary[0] = ((primary.len() as u32) << 16) | op;
        }
        if op == 57 && inst.len() >= 4 {
            if let Some(types) = functions
                .get(&inst[3])
                .and_then(|id| function_types.get(id))
            {
                if types.len() != inst.len() - 4 {
                    return Err(Error::Shader("invalid SPIR-V function call".into()));
                }
                primary.truncate(4);
                for (argument, ty) in inst[4..].iter().zip(types) {
                    primary.push(*argument);
                    if sampler_types.contains(ty) {
                        primary.push(clones.get(argument).copied().unwrap_or(*argument));
                    }
                }
                primary[0] = ((primary.len() as u32) << 16) | op;
            }
        }
        if op == 71 && inst.len() >= 4 && inst[2] == 33 {
            let set = sets.get(&inst[1]).copied().unwrap_or(0) as usize;
            if let Some((native, _)) = maps.get(set) {
                if let Some(&mapped) = native.get(&inst[3]) {
                    primary[3] = mapped;
                }
            }
        }
        if op == 86
            && inst.len() >= 5
            && (sampled
                .get(&inst[1])
                .and_then(|id| images.get(id))
                .is_some_and(|id| integer_types.contains(id)))
        {
            if let Some(&sampler) = clones.get(&inst[4]) {
                primary[4] = sampler;
            }
        }
        if op == 15 {
            // SPIR-V 1.4+ entry interfaces include all global variables used by
            // the entry. Include mirror variables alongside their originals.
            let mut name_end = 3;
            while name_end < inst.len() && !inst[name_end].to_le_bytes().contains(&0) {
                name_end += 1;
            }
            if name_end < inst.len() {
                let extras: Vec<_> = inst[name_end + 1..]
                    .iter()
                    .filter_map(|id| vars.contains_key(id).then(|| clones[id]))
                    .collect();
                primary.extend(extras);
                primary[0] = ((primary.len() as u32) << 16) | op;
            }
        }
        out.extend(primary);
        if op == 71 && inst.len() >= 3 {
            if let Some(&id) = clones.get(&inst[1]) {
                let mut copy = inst.to_vec();
                copy[1] = id;
                if copy.len() >= 4 && copy[2] == 33 {
                    copy[3] = vars[&inst[1]];
                }
                out.extend(copy);
            }
        }
        if op == 55 && inst.len() >= 3 {
            if let Some(&id) = clones.get(&inst[2]) {
                let mut copy = inst.to_vec();
                copy[2] = id;
                out.extend(copy);
            }
        } else if op == 59 && inst.len() >= 4 {
            if let Some(&id) = clones.get(&inst[2]) {
                let mut copy = inst.to_vec();
                copy[2] = id;
                out.extend(copy);
            }
        } else if matches!(op, 61 | 65 | 66 | 67 | 83) && inst.len() >= 4 {
            if let Some(&id) = clones.get(&inst[2]) {
                let mut copy = inst.to_vec();
                copy[2] = id;
                copy[3] = clones[&inst[3]];
                out.extend(copy);
            }
        } else if matches!(op, 169 | 245) && clones.contains_key(&inst[2]) {
            let mut copy = inst.to_vec();
            copy[2] = clones[&inst[2]];
            let positions: Vec<usize> = if op == 169 {
                vec![4, 5]
            } else {
                (3..inst.len()).step_by(2).collect()
            };
            for i in positions {
                copy[i] = clones.get(&inst[i]).copied().unwrap_or(inst[i]);
            }
            out.extend(copy);
        }
    }
    let info = vk::ShaderModuleCreateInfo::default().code(&out);
    // SAFETY: the words and live device outlive synchronous module creation.
    let module = unsafe { device.0.device.create_shader_module(&info, None) }.map_err(vk_error)?;
    Ok(VulkanShader(Arc::new(ShaderState {
        device: device.0.clone(),
        module,
        entries: shader.0.entries.clone(),
        words: out,
        wgsl: None,
    })))
}

/// Convert binding-bearing decoration groups to equivalent direct decorations.
fn flatten_binding_groups(words: &[u32]) -> Result<Vec<u32>> {
    let mut groups: HashMap<u32, Vec<Vec<u32>>> = HashMap::new();
    let mut cursor = 5;
    while cursor < words.len() {
        let count = (words[cursor] >> 16) as usize;
        if count == 0 || cursor + count > words.len() {
            return Err(Error::Shader("invalid SPIR-V instruction".into()));
        }
        let inst = &words[cursor..cursor + count];
        if inst[0] & 0xffff == 71 && count >= 3 {
            groups.entry(inst[1]).or_default().push(inst[2..].to_vec());
        }
        cursor += count;
    }
    let mut out = words[..5].to_vec();
    cursor = 5;
    while cursor < words.len() {
        let count = (words[cursor] >> 16) as usize;
        let inst = &words[cursor..cursor + count];
        if inst[0] & 0xffff == 74
            && count >= 3
            && groups
                .get(&inst[1])
                .is_some_and(|g| g.iter().any(|d| d[0] == 33 || d[0] == 34))
        {
            for target in &inst[2..] {
                for decoration in &groups[&inst[1]] {
                    out.push((((2 + decoration.len()) as u32) << 16) | 71);
                    out.push(*target);
                    out.extend(decoration);
                }
            }
        } else {
            out.extend_from_slice(inst);
        }
        cursor += count;
    }
    Ok(out)
}
/// Determine which public sampler bindings need integer and/or float sampling.
pub(crate) fn usage(
    shader: &VulkanShader,
    entry: &str,
    stage: api::ShaderStage,
    modes: &mut [HashMap<u32, u8>],
) -> Result<()> {
    let words = flatten_binding_groups(&shader.0.words)?;
    let mut instructions = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if n == 0 || at + n > words.len() {
            return Err(Error::Shader("invalid SPIR-V instruction".into()));
        }
        instructions.push(&words[at..at + n]);
        at += n;
    }
    let model = match stage {
        api::ShaderStage::Vertex => 0,
        api::ShaderStage::Fragment => 4,
        api::ShaderStage::Compute => 5,
        api::ShaderStage::Task => 5364,
        api::ShaderStage::Mesh => 5365,
        api::ShaderStage::RayGeneration => 5313,
        api::ShaderStage::Intersection => 5314,
        api::ShaderStage::AnyHit => 5315,
        api::ShaderStage::ClosestHit => 5316,
        api::ShaderStage::Miss => 5317,
        api::ShaderStage::Callable => 5318,
    };
    let mut root = None;
    let mut owners = Vec::with_capacity(instructions.len());
    let mut calls = Vec::new();
    let mut owner = 0;
    for inst in &instructions {
        let op = inst[0] & 0xffff;
        if op == 15 && inst.len() >= 4 {
            let name: Vec<u8> = inst[3..]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .take_while(|b| *b != 0)
                .collect();
            if inst[1] == model && name == entry.as_bytes() {
                root = Some(inst[2]);
            }
        }
        if op == 54 && inst.len() >= 3 {
            owner = inst[2];
        }
        owners.push(owner);
        if op == 57 && inst.len() >= 4 {
            calls.push((owner, inst[3]));
        }
        if op == 56 {
            owner = 0;
        }
    }
    let mut reachable = HashSet::from([root.ok_or(Error::InvalidState)?]);
    loop {
        let mut changed = false;
        for (caller, callee) in &calls {
            if reachable.contains(caller) {
                changed |= reachable.insert(*callee);
            }
        }
        if !changed {
            break;
        }
    }
    let mut binding = HashMap::new();
    let mut sets = HashMap::new();
    let mut integer = HashSet::new();
    let mut images = HashMap::new();
    let mut sampled = HashMap::new();
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut params: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut returns: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut current = 0;
    for (inst, &owner) in instructions.iter().zip(&owners) {
        let op = inst[0] & 0xffff;
        if owner != 0 && !reachable.contains(&owner) {
            continue;
        }
        match op {
            21 if inst.len() >= 2 => {
                integer.insert(inst[1]);
            }
            25 if inst.len() >= 3 => {
                images.insert(inst[1], inst[2]);
            }
            27 if inst.len() >= 3 => {
                sampled.insert(inst[1], inst[2]);
            }
            71 if inst.len() >= 4 && inst[2] == 33 => {
                binding.insert(inst[1], inst[3]);
            }
            71 if inst.len() >= 4 && inst[2] == 34 => {
                sets.insert(inst[1], inst[3]);
            }
            54 if inst.len() >= 3 => {
                current = inst[2];
            }
            55 if inst.len() >= 3 => {
                params.entry(current).or_default().push(inst[2]);
            }
            254 if inst.len() >= 2 => {
                returns.entry(current).or_default().push(inst[1]);
            }
            61 | 65 | 66 | 67 | 83 if inst.len() >= 4 => {
                edges.push((inst[3], inst[2]));
            }
            169 if inst.len() >= 6 => {
                edges.push((inst[4], inst[2]));
                edges.push((inst[5], inst[2]));
            }
            245 if inst.len() >= 5 => {
                for pair in inst[3..].chunks_exact(2) {
                    edges.push((pair[0], inst[2]));
                }
            }
            _ => {}
        }
    }
    for (inst, &owner) in instructions.iter().zip(&owners) {
        if owner != 0 && !reachable.contains(&owner) {
            continue;
        }
        if inst[0] & 0xffff == 57 && inst.len() >= 4 {
            if let Some(p) = params.get(&inst[3]) {
                for (&argument, &parameter) in inst[4..].iter().zip(p) {
                    edges.push((argument, parameter));
                }
            }
            if let Some(r) = returns.get(&inst[3]) {
                for &value in r {
                    edges.push((value, inst[2]));
                }
            }
        }
    }
    let mut origins: HashMap<u32, HashSet<u32>> = binding
        .keys()
        .map(|id| (*id, HashSet::from([*id])))
        .collect();
    loop {
        let mut changed = false;
        for (source, target) in &edges {
            let values = origins.get(source).cloned().unwrap_or_default();
            let dst = origins.entry(*target).or_default();
            let len = dst.len();
            dst.extend(values);
            changed |= len != dst.len();
        }
        if !changed {
            break;
        }
    }
    for (inst, &owner) in instructions.iter().zip(&owners) {
        if owner != 0 && !reachable.contains(&owner) {
            continue;
        }
        if inst[0] & 0xffff != 86 || inst.len() < 5 {
            continue;
        }
        let int = sampled
            .get(&inst[1])
            .and_then(|id| images.get(id))
            .is_some_and(|id| integer.contains(id));
        if origins.get(&inst[4]).is_none_or(|values| values.is_empty()) {
            return Err(Error::Shader(
                "sampler binding provenance cannot be determined".into(),
            ));
        }
        for id in origins.get(&inst[4]).into_iter().flatten() {
            let Some(&slot) = sets.get(id) else {
                continue;
            };
            let Some(&entry) = binding.get(id) else {
                continue;
            };
            let Some(group) = modes.get_mut(slot as usize) else {
                return Err(Error::InvalidState);
            };
            // Intermediate bitmask: bit 0 = float, bit 1 = integer.
            *group.entry(entry).or_default() |= if int { 2 } else { 1 };
        }
    }
    Ok(())
}
