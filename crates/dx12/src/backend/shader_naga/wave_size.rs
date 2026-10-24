//! Exact HLSL annotation for the selected generated compute entry.
//!
//! Naga does not model DX12's SM 6.6 `WaveSize` attribute.  This narrow
//! post-lowering pass only recognizes its compute-entry signature and inserts
//! a separate attribute line there; it never performs an unbounded text
//! replacement through generated HLSL.

use fluxel_rhi_core::api::{Error, Result};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1, ID3D12Device,
};

/// Wave builtins are usable only where D3D12 reports native WaveOps.
pub(crate) fn wave_ops_supported(device: &ID3D12Device) -> bool {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: `options` exactly matches the feature selector and is writable
    // for this synchronous capability query.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS1).cast(),
            std::mem::size_of_val(&options) as u32,
        )
    }
    .is_ok()
        && options.WaveOps.as_bool()
}

pub(crate) fn inject(source: &str, entry: &str, size: u32) -> Result<String> {
    if !size.is_power_of_two() {
        return Err(Error::InvalidRange);
    }
    let marker = format!("void {entry}(");
    let mut output = String::with_capacity(source.len() + 16);
    let mut inserted = false;
    for line in source.split_inclusive('\n') {
        if !inserted && line.trim_start().starts_with(&marker) {
            let indent = &line[..line.len() - line.trim_start().len()];
            output.push_str(indent);
            output.push_str("[WaveSize(");
            output.push_str(&size.to_string());
            output.push_str(")]\n");
            inserted = true;
        }
        output.push_str(line);
    }
    if !inserted {
        return Err(Error::Shader(
            "DX12 WGSL generated compute entry signature changed".into(),
        ));
    }
    Ok(output)
}

/// Inserts an attribute into the selected raw-HLSL function definition.
///
/// This bounded lexer deliberately handles only the source shape accepted by
/// the direct compute route: `void entry(...) { ... }`. It skips comments and
/// strings, ignores prototypes, and rejects preprocessing because a conditional
/// definition cannot be selected safely before DXC expands it.
pub(crate) fn inject_direct_hlsl(source: &str, entry: &str, size: u32) -> Result<String> {
    if !size.is_power_of_two() {
        return Err(Error::InvalidRange);
    }
    let tokens = lex(source)?;
    let declaration = top_level_entry_definition(source, &tokens, entry)?;
    if wave_attribute_before(source, &tokens, declaration)? {
        return Err(Error::Shader(
            "DX12 direct HLSL selected entry already declares WaveSize".into(),
        ));
    }
    let mut output = String::with_capacity(source.len() + 16);
    output.push_str(&source[..tokens[declaration].start]);
    output.push_str(&format!("[WaveSize({size})] "));
    output.push_str(&source[tokens[declaration].start..]);
    Ok(output)
}

fn top_level_entry_definition(source: &str, tokens: &[Token], entry: &str) -> Result<usize> {
    let mut brace_depth = 0u32;
    let mut found = None;
    for index in 0..tokens.len() {
        if tokens[index].punct(b'{') {
            brace_depth += 1;
            continue;
        }
        if tokens[index].punct(b'}') {
            brace_depth = brace_depth
                .checked_sub(1)
                .ok_or_else(|| Error::Shader("unbalanced HLSL braces".into()))?;
            continue;
        }
        if brace_depth != 0 {
            continue;
        }
        if index + 2 >= tokens.len()
            || !tokens[index].ident(source, "void")
            || !tokens[index + 1].ident(source, entry)
        {
            continue;
        }
        if !tokens[index + 2].punct(b'(') {
            continue;
        }
        let Some(close) = find_closing(&tokens, index + 2, b'(', b')') else {
            continue;
        };
        let mut after = close + 1;
        while after < tokens.len() && !tokens[after].punct(b'{') && !tokens[after].punct(b';') {
            after += 1;
        }
        if after == tokens.len() || tokens[after].punct(b';') {
            continue;
        }
        if found.replace(index).is_some() {
            return Err(Error::Shader(
                "DX12 direct HLSL has multiple top-level selected entry definitions".into(),
            ));
        }
    }
    if brace_depth != 0 {
        return Err(Error::Shader("unbalanced HLSL braces".into()));
    }
    found.ok_or_else(|| {
        Error::Shader(
            "DX12 direct HLSL required subgroup size needs a top-level void entry(...) definition"
                .into(),
        )
    })
}

#[derive(Clone, Copy)]
struct Token {
    start: usize,
    end: usize,
    punctuation: Option<u8>,
}
impl Token {
    fn ident(self, source: &str, expected: &str) -> bool {
        self.punctuation.is_none() && &source[self.start..self.end] == expected
    }
    fn punct(self, expected: u8) -> bool {
        self.punctuation == Some(expected)
    }
}

fn lex(source: &str) -> Result<Vec<Token>> {
    let bytes = source.as_bytes();
    let mut result = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at].is_ascii_whitespace() {
            at += 1;
        } else if bytes[at..].starts_with(b"//") {
            at = bytes[at..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |offset| at + offset + 1);
        } else if bytes[at..].starts_with(b"/*") {
            let begin = at + 2;
            let Some(offset) = bytes[begin..].windows(2).position(|part| part == b"*/") else {
                return Err(Error::Shader("unterminated HLSL block comment".into()));
            };
            at = begin + offset + 2;
        } else if matches!(bytes[at], b'\'' | b'\"') {
            let quote = bytes[at];
            at += 1;
            let mut closed = false;
            while at < bytes.len() {
                if bytes[at] == b'\\' {
                    at += 2;
                } else if bytes[at] == quote {
                    at += 1;
                    closed = true;
                    break;
                } else {
                    at += 1;
                }
            }
            if !closed {
                return Err(Error::Shader("unterminated HLSL string literal".into()));
            }
        } else if bytes[at] == b'#' {
            let end = bytes[at..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |offset| at + offset);
            let directive = std::str::from_utf8(&bytes[at + 1..end])
                .map_err(|_| Error::Shader("invalid HLSL directive".into()))?
                .trim_start();
            if !directive.starts_with("line") {
                return Err(Error::Unsupported(
                    "DX12 direct HLSL required subgroup size with unresolved preprocessor directive",
                ));
            }
            at = end.saturating_add(1);
        } else if bytes[at].is_ascii_alphabetic() || bytes[at] == b'_' {
            let start = at;
            at += 1;
            while at < bytes.len() && (bytes[at].is_ascii_alphanumeric() || bytes[at] == b'_') {
                at += 1;
            }
            result.push(Token {
                start,
                end: at,
                punctuation: None,
            });
        } else {
            result.push(Token {
                start: at,
                end: at + 1,
                punctuation: Some(bytes[at]),
            });
            at += 1;
        }
    }
    Ok(result)
}

fn find_closing(tokens: &[Token], open: usize, left: u8, right: u8) -> Option<usize> {
    let mut depth = 0;
    for (index, token) in tokens.iter().enumerate().skip(open) {
        if token.punct(left) {
            depth += 1;
        } else if token.punct(right) {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

fn wave_attribute_before(source: &str, tokens: &[Token], declaration: usize) -> Result<bool> {
    let mut end = declaration;
    while end > 0 && tokens[end - 1].punct(b']') {
        let Some(open) = find_opening(tokens, end - 1, b'[', b']') else {
            return Err(Error::Shader("unbalanced HLSL attribute".into()));
        };
        if tokens[open + 1..end - 1]
            .iter()
            .any(|token| token.ident(source, "WaveSize"))
        {
            return Ok(true);
        }
        end = open;
    }
    Ok(false)
}

fn find_opening(tokens: &[Token], close: usize, left: u8, right: u8) -> Option<usize> {
    let mut depth = 0;
    for index in (0..=close).rev() {
        if tokens[index].punct(right) {
            depth += 1;
        } else if tokens[index].punct(left) {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inserts_only_before_the_exact_compute_entry() {
        let source = "void helper() {}\nvoid main_() {}\nvoid main() {}\n";
        assert_eq!(
            inject(source, "main", 32).unwrap(),
            "void helper() {}\nvoid main_() {}\n[WaveSize(32)]\nvoid main() {}\n"
        );
    }

    #[test]
    fn rejects_missing_or_invalid_entry() {
        assert!(inject("void other() {}\n", "main", 32).is_err());
        assert!(inject("void main() {}\n", "main", 3).is_err());
    }

    #[test]
    fn direct_hlsl_finds_definition_not_comment_string_or_prototype() {
        let source = "// void main() {}\nconst char* note = \"void main() {}\";\nvoid main();\n[numthreads(1, 1, 1)] void main() {}";
        let result = inject_direct_hlsl(source, "main", 32).unwrap();
        assert!(result.contains("[numthreads(1, 1, 1)] [WaveSize(32)] void main()"));
        assert_eq!(result.matches("WaveSize").count(), 1);
    }

    #[test]
    fn direct_hlsl_handles_multiline_and_rejects_duplicate_or_preprocessor() {
        let multiline = "[numthreads(1, 1, 1)]\nvoid\nmain()\n{}";
        assert!(
            inject_direct_hlsl(multiline, "main", 32)
                .unwrap()
                .contains("[WaveSize(32)] void\nmain")
        );
        assert!(
            inject_direct_hlsl(
                "[WaveSize(16)] [numthreads(1,1,1)] void main() {}",
                "main",
                32
            )
            .is_err()
        );
        assert!(inject_direct_hlsl("#if 0\nvoid main() {}\n#endif", "main", 32).is_err());
    }

    #[test]
    fn direct_hlsl_ignores_method_decoy_and_rejects_multiple_global_entries() {
        let source = "struct Decoy { void main() {} };\nvoid main() {}";
        let result = inject_direct_hlsl(source, "main", 32).unwrap();
        assert!(result.contains("struct Decoy { void main() {} };\n[WaveSize(32)] void main()"));
        assert!(inject_direct_hlsl("void main() {}\nvoid main() {}", "main", 32).is_err());
    }
}
