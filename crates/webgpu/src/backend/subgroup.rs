//! Per-entry WGSL variants for WebGPU subgroup-size control.
//!
//! Shader source is kept in a weak map keyed by the browser module so retaining
//! source does not extend the lifetime of native shader modules.
use fluxel_rhi_core::api::{Error, Result};
use js_sys::{Object, WeakMap};
use wasm_bindgen::{JsCast, JsValue};

thread_local! {
    static SOURCES: WeakMap = WeakMap::new();
}

pub(super) fn remember(module: &JsValue, source: &str) {
    SOURCES.with(|sources| {
        sources.set(module.unchecked_ref::<Object>(), &JsValue::from_str(source));
    });
}

pub(super) fn source(module: &JsValue) -> Result<String> {
    SOURCES
        .with(|sources| sources.get(module.unchecked_ref::<Object>()).as_string())
        .ok_or_else(|| Error::Shader("WGSL source unavailable".into()))
}

#[derive(Clone, Copy)]
struct Token<'a> {
    text: &'a str,
    start: usize,
}

// WGSL comments can nest. Tokenize before identifying attributes to ensure
// apparent functions or attributes in comments can never affect the rewrite.
fn tokens(source: &str) -> Result<Vec<Token<'_>>> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if bytes[i..].starts_with(b"//") {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if bytes[i..].starts_with(b"/*") {
            i += 2;
            let mut depth = 1usize;
            while i < bytes.len() && depth != 0 {
                if bytes[i..].starts_with(b"/*") {
                    depth += 1;
                    i += 2;
                } else if bytes[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if depth != 0 {
                return Err(Error::Shader("unterminated WGSL comment".into()));
            }
            continue;
        }
        let start = i;
        if bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] >= 128 {
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] >= 128)
            {
                i += 1;
            }
        } else {
            i += 1;
        }
        let text = source
            .get(start..i)
            .ok_or_else(|| Error::Shader("invalid WGSL token".into()))?;
        tokens.push(Token { text, start });
    }
    Ok(tokens)
}

pub(super) fn variant(source: &str, entry: &str, size: u32) -> Result<String> {
    if !size.is_power_of_two() {
        return Err(Error::InvalidRange);
    }
    let tokens = tokens(source)?;
    let mut depth = 0usize;
    let mut boundary = 0usize;
    for (i, token) in tokens.iter().enumerate() {
        match token.text {
            "{" => {
                depth += 1;
            }
            "}" => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| Error::Shader("unbalanced WGSL braces".into()))?;
                if depth == 0 {
                    boundary = i + 1;
                }
            }
            ";" if depth == 0 => boundary = i + 1,
            "fn" if depth == 0 && tokens.get(i + 1).is_some_and(|v| v.text == entry) => {
                let attrs = &tokens[boundary..i];
                if !attrs
                    .windows(2)
                    .any(|w| w[0].text == "@" && w[1].text == "compute")
                {
                    return Err(Error::Shader(
                        "subgroup size requires a compute entry point".into(),
                    ));
                }
                if let Some(pos) = attrs
                    .windows(2)
                    .position(|w| w[0].text == "@" && w[1].text == "subgroup_size")
                {
                    let begin = boundary + pos;
                    if !tokens.get(begin + 2).is_some_and(|t| t.text == "(") {
                        return Err(Error::Shader("invalid subgroup_size attribute".into()));
                    }
                    let mut nesting = 0usize;
                    let mut close = None;
                    for t in &tokens[begin + 2..i] {
                        match t.text {
                            "(" => nesting += 1,
                            ")" => {
                                nesting -= 1;
                                if nesting == 0 {
                                    close = Some(t.start + 1);
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                    let close =
                        close.ok_or_else(|| Error::Shader("unterminated subgroup_size".into()))?;
                    let prefix = if tokens
                        .windows(2)
                        .any(|v| v[0].text == "enable" && v[1].text == "subgroup_size_control")
                    {
                        ""
                    } else {
                        "enable subgroup_size_control;\n"
                    };
                    return Ok(format!(
                        "{prefix}{}@subgroup_size({size}){}",
                        &source[..tokens[begin].start],
                        &source[close..]
                    ));
                }
                let at = token.start;
                let prefix = if tokens
                    .windows(2)
                    .any(|v| v[0].text == "enable" && v[1].text == "subgroup_size_control")
                {
                    ""
                } else {
                    "enable subgroup_size_control;\n"
                };
                return Ok(format!(
                    "{prefix}{}@subgroup_size({size}) {}",
                    &source[..at],
                    &source[at..]
                ));
            }
            _ => {}
        }
    }
    Err(Error::Shader(format!(
        "WGSL compute entry point '{entry}' not found"
    )))
}
