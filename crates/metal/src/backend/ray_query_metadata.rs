//! Propagate acceleration-structure metadata through Naga-generated MSL calls.
//!
//! Each structure parameter has an accompanying device pointer. Entry-point
//! pointers and their native slots are installed by the shader stage ABI.

use fluxel_rhi_core::api::{Error, Result};

const AS_TYPE: &str = "metal::raytracing::instance_acceleration_structure";

struct Function {
    name: String,
    body: std::ops::Range<usize>,
    parameters: Vec<usize>,
}

fn invalid() -> Error {
    Error::Shader("unsupported generated Metal ray-query metadata ABI".into())
}

fn ident(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn matching(source: &str, open: usize, left: u8, right: u8) -> Result<usize> {
    let mut depth = 0usize;
    for (i, &c) in source.as_bytes().iter().enumerate().skip(open) {
        if c == left {
            depth += 1;
        } else if c == right {
            depth = depth.checked_sub(1).ok_or_else(invalid)?;
            if depth == 0 {
                return Ok(i);
            }
        }
    }
    Err(invalid())
}

fn arguments(
    source: &str,
    start: usize,
    end: usize,
    declaration: bool,
) -> Result<Vec<std::ops::Range<usize>>> {
    let mut result = Vec::new();
    let mut begin = start;
    let mut parens = 0usize;
    let mut brackets = 0usize;
    let mut angles = 0usize;
    for i in start..end {
        match source.as_bytes()[i] {
            b'(' => parens += 1,
            b')' => parens = parens.checked_sub(1).ok_or_else(invalid)?,
            b'[' => brackets += 1,
            b']' => brackets = brackets.checked_sub(1).ok_or_else(invalid)?,
            b'<' if declaration && parens == 0 && brackets == 0 => angles += 1,
            b'>' if parens == 0 && brackets == 0 && angles != 0 => angles -= 1,
            b',' if parens == 0 && brackets == 0 && angles == 0 => {
                result.push(begin..i);
                begin = i + 1;
            }
            _ => {}
        }
    }
    if parens != 0 || brackets != 0 || angles != 0 {
        return Err(invalid());
    }
    if !source[begin..end].trim().is_empty() {
        result.push(begin..end);
    }
    Ok(result)
}

pub(super) fn metadata_name(expression: &str, suffix: &str) -> Result<String> {
    let expression = expression.trim();
    let mut depth = 0usize;
    for (position, byte) in expression.bytes().enumerate() {
        match byte {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth = depth.checked_sub(1).ok_or_else(invalid)?,
            b'?' if depth == 0 => {
                let remainder = &expression[position + 1..];
                let marker = " : DefaultConstructible()";
                let value = remainder.strip_suffix(marker).ok_or_else(invalid)?;
                let value = metadata_name(value, suffix)?;
                return Ok(format!(
                    "({} ? {value} : nullptr)",
                    expression[..position].trim()
                ));
            }
            _ => {}
        }
    }
    if expression.starts_with('(') && matching(expression, 0, b'(', b')')? == expression.len() - 1 {
        return metadata_name(&expression[1..expression.len() - 1], suffix);
    }
    if let Some(open) = expression.find('[') {
        let base = metadata_name(&expression[..open], suffix)?;
        let close = matching(expression, open, b'[', b']')?;
        if expression[close + 1..].trim() != ".inner" {
            return Err(invalid());
        }
        let index = expression[open + 1..close].trim();
        if index.is_empty() {
            return Err(invalid());
        }
        return Ok(format!(
            "({base}[{index}] == 0u ? nullptr : {base} + {base}[{index}])"
        ));
    }
    if expression.is_empty()
        || !expression.bytes().all(ident)
        || expression.as_bytes()[0].is_ascii_digit()
    {
        return Err(invalid());
    }
    Ok(format!("{expression}{suffix}"))
}

/// Add metadata parameters and propagate corresponding arguments at calls.
pub(super) fn propagate(source: &str, suffix: &str) -> Result<String> {
    let bytes = source.as_bytes();
    let mut functions = Vec::new();
    let mut edits: Vec<(usize, String)> = Vec::new();
    let mut i = 0;
    let mut braces = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => braces += 1,
            b'}' => braces = braces.checked_sub(1).ok_or_else(invalid)?,
            b'(' if braces == 0 => {
                let close = matching(source, i, b'(', b')')?;
                let mut after = close + 1;
                while after < bytes.len() && bytes[after].is_ascii_whitespace() {
                    after += 1;
                }
                if after < bytes.len() && bytes[after] == b'{' {
                    let mut name_end = i;
                    while name_end > 0 && bytes[name_end - 1].is_ascii_whitespace() {
                        name_end -= 1;
                    }
                    let mut name_start = name_end;
                    while name_start > 0 && ident(bytes[name_start - 1]) {
                        name_start -= 1;
                    }
                    if name_start == name_end {
                        return Err(invalid());
                    }
                    let args = arguments(source, i + 1, close, true)?;
                    let mut parameters = Vec::new();
                    for (index, range) in args.iter().enumerate() {
                        let arg = source[range.clone()].trim();
                        if let Some(position) = arg.find(AS_TYPE) {
                            let tail = &arg[position + AS_TYPE.len()..];
                            let tail = tail.trim_start_matches(|c: char| {
                                c == '>' || c == '*' || c == '&' || c.is_ascii_whitespace()
                            });
                            let name = tail.split_whitespace().next().ok_or_else(invalid)?;
                            let metadata = metadata_name(name, suffix)?;
                            if !arg.contains("[[buffer(") {
                                edits.push((range.end, format!(", const device uint* {metadata}")));
                                parameters.push(index);
                            }
                        }
                    }
                    let end = matching(source, after, b'{', b'}')?;
                    functions.push(Function {
                        name: source[name_start..name_end].to_owned(),
                        body: after + 1..end,
                        parameters,
                    });
                    i = end + 1;
                    continue;
                }
                i = close;
            }
            _ => {}
        }
        i += 1;
    }
    for caller in &functions {
        let mut cursor = caller.body.start;
        while cursor < caller.body.end {
            if bytes[cursor].is_ascii_alphabetic() || bytes[cursor] == b'_' {
                let start = cursor;
                cursor += 1;
                while cursor < caller.body.end && ident(bytes[cursor]) {
                    cursor += 1;
                }
                let name = &source[start..cursor];
                let Some(callee) = functions
                    .iter()
                    .find(|f| f.name == name && !f.parameters.is_empty())
                else {
                    continue;
                };
                let mut open = cursor;
                while open < caller.body.end && bytes[open].is_ascii_whitespace() {
                    open += 1;
                }
                if open >= caller.body.end || bytes[open] != b'(' {
                    continue;
                }
                let close = matching(source, open, b'(', b')')?;
                let args = arguments(source, open + 1, close, false)?;
                for &index in &callee.parameters {
                    let arg = args.get(index).ok_or_else(invalid)?;
                    edits.push((
                        arg.end,
                        format!(", {}", metadata_name(&source[arg.clone()], suffix)?),
                    ));
                }
            } else {
                cursor += 1;
            }
        }
    }
    edits.sort_by_key(|(position, _)| std::cmp::Reverse(*position));
    let mut result = source.to_owned();
    for (position, value) in edits {
        result.insert_str(position, &value);
    }
    Ok(result)
}
