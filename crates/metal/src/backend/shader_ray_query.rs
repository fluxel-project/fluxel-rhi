//! Complete Naga 30.0.1's Metal ray-query lowering.
//!
//! Naga emits native intersection queries but omits triangle-face and
//! primitive-type culling and the instance shader-table contribution. The
//! latter is carried alongside each acceleration structure by the stage ABI.
//! Generated constructs are checked before rewriting so a writer change is a
//! shader error rather than a silently incorrect intersection.

use fluxel_rhi_core::api::{Error, Result};

#[path = "ray_query_metadata.rs"]
mod metadata;

pub(super) fn lower(mut source: String, metadata_slots: &[(u8, u8)]) -> Result<String> {
    let mut unique = 0usize;
    let (suffix, query_prefix, helper_metadata) = loop {
        let suffix = format!("_fluxel_metadata_{unique}");
        let query = format!("_fluxel_query_metadata_{unique}_");
        let helper = format!("_fluxel_helper_metadata_{unique}");
        if !source.contains(&suffix) && !source.contains(&query) && !source.contains(&helper) {
            break (suffix, query, helper);
        }
        unique = unique.checked_add(1).ok_or(Error::InvalidRange)?;
    };
    for &(as_slot, metadata_slot) in metadata_slots {
        let attribute = format!("[[buffer({as_slot})]]");
        let attr = source
            .find(&attribute)
            .ok_or_else(|| Error::Shader("Metal acceleration-structure argument missing".into()))?;
        let prefix = &source[..attr];
        let end = prefix.trim_end().len();
        let start = prefix[..end]
            .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .map_or(0, |i| i + 1);
        let name = &source[start..end];
        if !identifier(name) {
            return Err(Error::Shader(
                "invalid Metal acceleration-structure argument".into(),
            ));
        }
        let declaration = prefix[..start].rsplit([',', '(']).next().unwrap_or("");
        if !declaration.contains("metal::raytracing::instance_acceleration_structure") {
            return Err(Error::Shader(
                "unexpected Metal acceleration-structure ABI".into(),
            ));
        }
        let argument = format!(", const device uint* {name}{suffix} [[buffer({metadata_slot})]]");
        source.insert_str(attr + attribute.len(), &argument);
    }
    const MARKER: &str = "params.accept_any_intersection((desc.flags & 4) != 0);";
    const REPLACEMENT: &str = concat!(
        "if ((desc.flags & 48) == 16) params.set_triangle_cull_mode(metal::raytracing::triangle_cull_mode::back);\n",
        "if ((desc.flags & 48) == 32) params.set_triangle_cull_mode(metal::raytracing::triangle_cull_mode::front);\n",
        "if ((desc.flags & 256) != 0 || (desc.flags & 48) == 48) params.set_geometry_cull_mode(metal::raytracing::geometry_cull_mode::triangle);\n",
        "else if ((desc.flags & 512) != 0) params.set_geometry_cull_mode(metal::raytracing::geometry_cull_mode::bounding_box);\n",
        "params.accept_any_intersection((desc.flags & 4) != 0);"
    );
    let initializers = source
        .matches("metal::raytracing::intersection_params params;")
        .count();
    if source.matches(MARKER).count() != initializers {
        return Err(Error::Shader(
            "Metal ray-query initialization does not match Naga 30.0.1".into(),
        ));
    }
    let mut source = metadata::propagate(&source.replace(MARKER, REPLACEMENT), &suffix)?;
    for kind in ["false", "true"] {
        let marker =
            format!("ray_query_get_intersection_{kind}(metal::raytracing::intersection_query<");
        if let Some(start) = source.find(&marker) {
            let end = source[start..].find(')').ok_or_else(|| {
                Error::Shader("unexpected Metal ray-query helper declaration".into())
            })? + start;
            source.insert_str(end, &format!(", const device uint* {helper_metadata}"));
        }
    }
    let mut output = String::with_capacity(source.len());
    let mut pending_reset: Option<(String, u8)> = None;
    for line in source.split_inclusive('\n') {
        let text = line.trim();
        if let Some((query, remaining)) = pending_reset.as_mut() {
            let expected = if *remaining == 2 {
                format!("naga_query_init_tracker_for_{query} = ")
            } else {
                format!("naga_query_tmax_tracker_for_{query} = ")
            };
            if !text.starts_with(&expected) {
                return Err(Error::Shader(
                    "unexpected Metal ray-query tracking sequence".into(),
                ));
            }
            output.push_str(line);
            *remaining -= 1;
            if *remaining == 0 {
                output.push_str(&format!(
                    "    }} else {{ naga_query_init_tracker_for_{query} = 0u; }}\n"
                ));
                pending_reset = None;
            }
            continue;
        }
        const AS: &str = "metal::raytracing::instance_acceleration_structure ";
        if let Some(rest) = text.strip_prefix(AS) {
            if let Some((name, expression)) =
                rest.strip_suffix(';').and_then(|s| s.split_once(" = "))
            {
                if !identifier(name.trim()) {
                    return Err(Error::Shader(
                        "unexpected baked Metal acceleration structure".into(),
                    ));
                }
                let metadata = metadata::metadata_name(expression, &suffix)?;
                output.push_str(line);
                output.push_str(&format!(
                    "    const device uint* {}{suffix} = {metadata};\n",
                    name.trim()
                ));
                continue;
            }
        }
        const QUERY: &str = "metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> ";
        if let Some(rest) = text.strip_prefix(QUERY) {
            let name = rest.split_whitespace().next().unwrap_or("");
            if !identifier(name) || !rest.ends_with("= {};") {
                return Err(Error::Shader(
                    "unexpected Metal ray-query declaration".into(),
                ));
            }
            output.push_str(line);
            output.push_str(&format!(
                "    const device uint* {query_prefix}{name} = nullptr;\n"
            ));
            continue;
        }
        if let Some((before, rest)) = text.split_once(".reset(ray,") {
            let Some((accel, _)) = rest.rsplit_once(", desc.cull_mask, params);") else {
                return Err(Error::Shader("unexpected Metal ray-query reset".into()));
            };
            if !identifier(before) {
                return Err(Error::Shader(
                    "unsupported Metal ray-query acceleration expression".into(),
                ));
            }
            let metadata = metadata::metadata_name(accel, &suffix)?;
            output.push_str(&format!("    {query_prefix}{before} = {metadata};\n"));
            output.push_str(&format!("    if ({query_prefix}{before} != nullptr) {{\n"));
            pending_reset = Some((before.to_owned(), 2));
        }
        let mut rewritten = line.to_owned();
        if text.contains(".reset(ray,") {
            rewritten = rewritten.replace(", desc.cull_mask, params);", ", ((((desc.flags & 192) == 192) || ((desc.flags & 512) != 0 && ((desc.flags & 256) != 0 || (desc.flags & 48) == 48))) ? 0u : desc.cull_mask), params);");
        }
        for kind in ["false", "true"] {
            let helper = format!("ray_query_get_intersection_{kind}(");
            let mut cursor = 0;
            while let Some(relative) = rewritten[cursor..].find(&helper) {
                let pos = cursor + relative;
                let start = pos + helper.len();
                if rewritten[start..].starts_with("metal::raytracing::intersection_query<") {
                    break;
                }
                let end = rewritten[start..]
                    .find(')')
                    .ok_or_else(|| Error::Shader("unexpected Metal ray-query helper".into()))?
                    + start;
                let args = &rewritten[start..end];
                let name = args.split(',').next().unwrap_or("").trim();
                if !identifier(name) {
                    return Err(Error::Shader("unexpected Metal ray-query argument".into()));
                }
                let argument = format!(", {query_prefix}{name}");
                rewritten.insert_str(end, &argument);
                cursor = end + argument.len() + 1;
            }
        }
        if text.contains("intersection.instance_index = intersector.get_") {
            output.push_str(&rewritten);
            output.push_str(&format!("        intersection.sbt_record_offset = {helper_metadata}[intersection.instance_index];\n"));
            continue;
        }
        output.push_str(&rewritten);
    }
    if pending_reset.is_some() {
        return Err(Error::Shader(
            "incomplete Metal ray-query tracking sequence".into(),
        ));
    }
    Ok(output)
}

fn identifier(value: &str) -> bool {
    let mut chars = value.bytes();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == b'_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == b'_')
}
