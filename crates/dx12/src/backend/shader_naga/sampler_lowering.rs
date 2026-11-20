//! Lowers Naga sampler heap references to DX12 descriptor tables.

use fluxel_rhi_core::api::{Error, Result};

pub(super) fn lower_sampler_arrays(
    source: &str,
    arrays: &[(u32, u32, bool, u32, String)],
) -> Result<String> {
    let mut output = source.to_owned();
    for (group, binding, comparison, count, marker) in arrays {
        let prefix = format!("static const uint {marker}");
        let line = output
            .lines()
            .find(|line| line.starts_with(&prefix))
            .ok_or(Error::Unsupported("DX12 WGSL sampler array declaration"))?;
        let (left, value) = line
            .split_once(" = ")
            .ok_or(Error::Unsupported("DX12 WGSL sampler array declaration"))?;
        if value != format!("{binding};") {
            return Err(Error::Unsupported("DX12 WGSL sampler array binding"));
        }
        let name = left
            .strip_prefix("static const uint ")
            .ok_or(Error::InvalidState)?
            .to_owned();
        let replacement = format!(
            "{} {name}[{count}] : register(s{binding}, space{group});",
            if *comparison {
                "SamplerComparisonState"
            } else {
                "SamplerState"
            }
        );
        output = output.replacen(line, &replacement, 1);
        let heap = if *comparison {
            "nagaComparisonSamplerHeap"
        } else {
            "nagaSamplerHeap"
        };
        for nonuniform in [false, true] {
            let prefix = if nonuniform {
                format!(
                    "{heap}[NonUniformResourceIndex(nagaGroup{group}SamplerIndexArray[{name} + "
                )
            } else {
                format!("{heap}[nagaGroup{group}SamplerIndexArray[{name} + ")
            };
            while let Some(start) = output.find(&prefix) {
                let expr_start = start + prefix.len();
                let mut depth = 0usize;
                let mut end = None;
                for (offset, ch) in output[expr_start..].char_indices() {
                    match ch {
                        '[' => depth += 1,
                        ']' if depth == 0 => {
                            end = Some(expr_start + offset);
                            break;
                        }
                        ']' => depth -= 1,
                        _ => {}
                    }
                }
                let end = end.ok_or(Error::Unsupported("DX12 WGSL sampler array expression"))?;
                let suffix = if nonuniform { "])]" } else { "]]" };
                if !output[end..].starts_with(suffix) {
                    return Err(Error::Unsupported("DX12 WGSL sampler array expression"));
                }
                let expr = &output[expr_start..end];
                let replacement = if nonuniform {
                    format!("{name}[NonUniformResourceIndex({expr})]")
                } else {
                    format!("{name}[{expr}]")
                };
                output.replace_range(start..end + suffix.len(), &replacement);
            }
        }
    }
    Ok(output)
}

/// Naga 27 emits a sampler heap plus an index buffer even for scalar samplers.
/// Rewrite only its exact, pinned scalar declaration form into root-table slots;
/// reject any unexpected generated form instead of silently changing the ABI.
pub(super) fn lower_scalar_samplers(
    source: &str,
    samplers: &[(u32, u32, bool)],
    groups: &[u32],
) -> Result<String> {
    let mut output = String::new();
    let mut seen = Vec::new();
    for line in source.lines() {
        if line == "SamplerState nagaSamplerHeap[2048]: register(s0, space0);"
            || line
                == "SamplerComparisonState nagaComparisonSamplerHeap[2048]: register(s0, space1);"
        {
            continue;
        }
        let mut index_decl = false;
        for &group in groups {
            if line
                == format!(
                    "StructuredBuffer<uint> nagaGroup{group}SamplerIndexArray : register(t{group}, space255);"
                )
            {
                index_decl = true;
            }
        }
        if index_decl {
            continue;
        }
        let declaration = line
            .strip_prefix("static const SamplerState ")
            .map(|rest| (rest, false))
            .or_else(|| {
                line.strip_prefix("static const SamplerComparisonState ")
                    .map(|rest| (rest, true))
            });
        if let Some((rest, comparison)) = declaration {
            let (name, expr) = rest
                .split_once(" = ")
                .ok_or(Error::Unsupported("DX12 WGSL sampler lowering"))?;
            let mut matched = None;
            for &(group, binding, expected_comparison) in samplers {
                let heap = if comparison {
                    "nagaComparisonSamplerHeap"
                } else {
                    "nagaSamplerHeap"
                };
                if comparison == expected_comparison
                    && expr == format!("{heap}[nagaGroup{group}SamplerIndexArray[{binding}]];")
                {
                    matched = Some((group, binding));
                    break;
                }
            }
            let (group, binding) =
                matched.ok_or(Error::Unsupported("DX12 WGSL sampler lowering"))?;
            output.push_str(&format!(
                "{} {name} : register(s{binding}, space{group});\n",
                if comparison {
                    "SamplerComparisonState"
                } else {
                    "SamplerState"
                }
            ));
            seen.push((group, binding, comparison));
            continue;
        }
        output.push_str(line);
        output.push('\n');
    }
    if (source.contains("nagaSamplerHeap") || source.contains("nagaComparisonSamplerHeap"))
        && (samplers.iter().any(|item| !seen.contains(item))
            || output.contains("nagaSamplerHeap")
            || output.contains("nagaComparisonSamplerHeap")
            || output.contains("SamplerIndexArray"))
    {
        return Err(Error::Unsupported("DX12 WGSL sampler lowering"));
    }
    Ok(output)
}
