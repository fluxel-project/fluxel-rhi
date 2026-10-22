//! WGSL immediate-data ABI translation for per-stage byte offsets.
//!
//! The returned source remains WGSL, allowing further transformations such as
//! subgroup-size variants before creating the native shader module.
use fluxel_rhi_core::api::{Error, Result};

#[derive(Clone, Copy)]
struct Token<'a> {
    text: &'a str,
    start: usize,
    end: usize,
}

fn fail(message: &str) -> Error {
    Error::Shader(format!("WebGPU immediate ABI: {message}"))
}

fn tokenize(source: &str) -> Result<Vec<Token<'_>>> {
    let bytes = source.as_bytes();
    let mut result = Vec::new();
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
                return Err(fail("unterminated block comment"));
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
        let text = source.get(start..i).ok_or_else(|| fail("invalid token"))?;
        result.push(Token {
            text,
            start,
            end: i,
        });
    }
    Ok(result)
}

/// Returns whether the module declares immediate data, ignoring comments.
pub(super) fn has_immediate(source: &str) -> Result<bool> {
    Ok(!declarations(&tokenize(source)?)?.is_empty())
}

/// Translate immediate globals using Naga name resolution. The parser input is
/// position-preserving so each resolved global reference can be edited in the
/// original source without changing comments, member names, or shadowed locals.
pub(super) fn variant(source: &str, offset: u32, entry: &str) -> Result<String> {
    if offset % 4 != 0 {
        return Err(fail("offset must be four-byte aligned"));
    }
    let tokens = tokenize(source)?;
    let declarations = declarations(&tokens)?;
    if declarations.is_empty() {
        return Ok(source.to_owned());
    }
    let module = parse(source, &tokens, &declarations)?;
    let has_arrays = declarations.iter().any(|&(start, _)| {
        module
            .global_variables
            .iter()
            .find(|(_, g)| g.name.as_deref() == Some(tokens[start + 4].text))
            .is_some_and(|(_, g)| contains_array(&module, g.ty))
    });
    if has_arrays || has_immediate_pointer(&tokens) {
        return flatten_arrays(source, offset, entry, &tokens, &declarations, &module);
    }
    let mut edits = Vec::<(usize, usize, String)>::new();
    if offset == 0 {
        for &(start, _) in &declarations {
            edits.push((
                tokens[start + 2].start,
                tokens[start + 2].end,
                "immediate".into(),
            ));
        }
    } else {
        let module = parse(source, &tokens, &declarations)?;
        let mut layouts = naga::proc::Layouter::default();
        layouts
            .update(module.to_ctx())
            .map_err(|e| Error::Shader(format!("WebGPU immediate ABI layout: {e}")))?;
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| Error::Shader(format!("WebGPU immediate ABI validation: {e}")))?;
        let entry_index = module
            .entry_points
            .iter()
            .position(|ep| ep.name == entry)
            .ok_or_else(|| fail("pipeline entry not found"))?;
        let mut globals = Vec::new();
        for (index, &(start, end)) in declarations.iter().enumerate() {
            let name = tokens[start + 4].text;
            let (handle, global) = module
                .global_variables
                .iter()
                .find(|(_, g)| g.name.as_deref() == Some(name))
                .ok_or_else(|| fail("global name resolution failed"))?;
            let alignment = type_alignment(&module, global.ty, source, &layouts)?.max(4);
            let active = !info.get_entry_point(entry_index)[handle].is_empty();
            if active && offset % alignment != 0 {
                return Err(fail("offset does not satisfy immediate type alignment"));
            }
            let actual_offset = if active {
                offset
            } else {
                round_up(offset, alignment)?
            };
            let wrapper = (0u32..)
                .map(|n| format!("_fluxel_immediate_wrapper_{index}_{n}"))
                .find(|name| !tokens.iter().any(|t| t.text == name))
                .ok_or_else(|| fail("cannot generate identifier"))?;
            let ty = &source[tokens[start + 6].start..tokens[end].start];
            edits.push((tokens[start].start, tokens[end].end, format!("struct {wrapper} {{ @size({actual_offset}) _fluxel_pad: u32, _fluxel_value: {ty}, }};\nvar<immediate> {name}: {wrapper};")));
            globals.push((handle, name));
        }
        for function in module
            .functions
            .iter()
            .map(|(_, f)| f)
            .chain(module.entry_points.iter().map(|ep| &ep.function))
        {
            for (handle, expression) in function.expressions.iter() {
                if let naga::Expression::GlobalVariable(global) = expression {
                    if let Some((_, name)) = globals.iter().find(|(target, _)| target == global) {
                        let span = function
                            .expressions
                            .get_span(handle)
                            .to_range()
                            .ok_or_else(|| fail("global reference has no source span"))?;
                        if source.get(span.clone()) != Some(*name) {
                            return Err(fail("global reference span is ambiguous"));
                        }
                        edits.push((span.end, span.end, "._fluxel_value".into()));
                    }
                }
            }
        }
    }
    edits.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    edits.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1 && a.2 == b.2);
    let mut output = source.to_owned();
    for (start, end, replacement) in edits {
        output.replace_range(start..end, &replacement);
    }
    let already_required = tokens.iter().enumerate().any(|(i, token)| {
        token.text == "requires"
            && tokens[i + 1..]
                .iter()
                .take_while(|token| token.text != ";")
                .any(|token| token.text == "immediate_address_space")
    });
    if already_required {
        Ok(output)
    } else {
        Ok(format!("requires immediate_address_space;\n{output}"))
    }
}

fn declarations(tokens: &[Token<'_>]) -> Result<Vec<(usize, usize)>> {
    let mut depth = 0usize;
    let mut declarations = Vec::new();
    for (i, token) in tokens.iter().enumerate() {
        match token.text {
            "{" => depth += 1,
            "}" => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| fail("unbalanced braces"))?
            }
            "var"
                if depth == 0
                    && tokens.get(i + 1).is_some_and(|t| t.text == "<")
                    && tokens
                        .get(i + 2)
                        .is_some_and(|t| matches!(t.text, "push_constant" | "immediate")) =>
            {
                if !tokens.get(i + 3).is_some_and(|t| t.text == ">")
                    || !tokens.get(i + 5).is_some_and(|t| t.text == ":")
                {
                    return Err(fail("unsupported immediate declaration"));
                }
                let end = (i + 6..tokens.len())
                    .find(|&j| tokens[j].text == ";")
                    .ok_or_else(|| fail("unterminated immediate declaration"))?;
                if end == i + 6 || tokens[i + 6..end].iter().any(|t| t.text == "=") {
                    return Err(fail("invalid immediate declaration"));
                }
                declarations.push((i, end));
            }
            _ => {}
        }
    }
    Ok(declarations)
}

fn parse(
    source: &str,
    tokens: &[Token<'_>],
    declarations: &[(usize, usize)],
) -> Result<naga::Module> {
    let mut input = source.as_bytes().to_vec();
    for &(start, _) in declarations {
        let t = tokens[start + 2];
        input[t.start..t.end].fill(b' ');
        input[t.start..t.start + 7].copy_from_slice(b"private");
    }
    // Immediate pointers in helper signatures must agree with the temporary
    // private globals used for name resolution and array reconstruction.
    for window in tokens.windows(3) {
        if window[0].text == "ptr"
            && window[1].text == "<"
            && matches!(window[2].text, "immediate" | "push_constant")
        {
            let t = window[2];
            input[t.start..t.end].fill(b' ');
            input[t.start..t.start + 7].copy_from_slice(b"private");
        }
    }
    // Naga does not yet implement all WGSL directives and the subgroup attribute. Mask
    // them without changing byte offsets used by expression spans.
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i].text == "requires" {
            let end = (i + 1..tokens.len())
                .find(|&j| tokens[j].text == ";")
                .ok_or_else(|| fail("unterminated requires directive"))?;
            input[tokens[i].start..tokens[end].end].fill(b' ');
            i = end + 1;
            continue;
        }
        if tokens[i].text == "subgroup_size_control"
            && tokens[..i]
                .iter()
                .rev()
                .take_while(|t| t.text != ";")
                .any(|t| t.text == "enable")
        {
            let t = tokens[i];
            input[t.start..t.end].fill(b' ');
            input[t.start..t.start + 3].copy_from_slice(b"f16");
        }
        if tokens[i].text == "@" && tokens.get(i + 1).is_some_and(|t| t.text == "subgroup_size") {
            if !tokens.get(i + 2).is_some_and(|t| t.text == "(") {
                return Err(fail("invalid subgroup attribute"));
            }
            let mut level = 1usize;
            let mut end = i + 3;
            while end < tokens.len() && level != 0 {
                if tokens[end].text == "(" {
                    level += 1;
                }
                if tokens[end].text == ")" {
                    level -= 1;
                }
                end += 1;
            }
            if level != 0 {
                return Err(fail("unterminated subgroup attribute"));
            }
            input[tokens[i].start..tokens[end - 1].end].fill(b' ');
            i = end;
            continue;
        }
        i += 1;
    }
    let input = std::str::from_utf8(&input).map_err(|_| fail("invalid WGSL preprocessing"))?;
    naga::front::wgsl::parse_str(input)
        .map_err(|e| Error::Shader(format!("WebGPU immediate ABI name resolution: {e}")))
}

/// Required byte alignment for the immediate types declared by a shader.
pub(super) fn required_alignment(source: &str, entry: &str) -> Result<u32> {
    let tokens = tokenize(source)?;
    let declarations = declarations(&tokens)?;
    if declarations.is_empty() {
        return Ok(4);
    }
    let module = parse(source, &tokens, &declarations)?;
    let mut layouts = naga::proc::Layouter::default();
    layouts
        .update(module.to_ctx())
        .map_err(|e| Error::Shader(format!("WebGPU immediate ABI layout: {e}")))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| Error::Shader(format!("WebGPU immediate ABI validation: {e}")))?;
    let entry_index = module
        .entry_points
        .iter()
        .position(|ep| ep.name == entry)
        .ok_or_else(|| fail("pipeline entry not found"))?;
    // Array lowering uses u32-only immediate structs for all globals.
    if has_immediate_pointer(&tokens)
        || declarations.iter().any(|&(start, _)| {
            module
                .global_variables
                .iter()
                .find(|(_, g)| g.name.as_deref() == Some(tokens[start + 4].text))
                .is_some_and(|(_, g)| contains_array(&module, g.ty))
        })
    {
        return Ok(4);
    }
    let mut alignment = 4;
    for (start, _) in declarations {
        let name = tokens[start + 4].text;
        let (handle, global) = module
            .global_variables
            .iter()
            .find(|(_, g)| g.name.as_deref() == Some(name))
            .ok_or_else(|| fail("global name resolution failed"))?;
        if !info.get_entry_point(entry_index)[handle].is_empty() {
            alignment = alignment.max(type_alignment(&module, global.ty, source, &layouts)?);
        }
    }
    Ok(alignment)
}

fn type_alignment(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    source: &str,
    layouts: &naga::proc::Layouter,
) -> Result<u32> {
    let mut alignment = layouts[ty]
        .alignment
        .to_string()
        .parse::<u32>()
        .map_err(|_| fail("invalid type alignment"))?;
    match &module.types[ty].inner {
        naga::TypeInner::Struct { members, .. } => {
            let span = module
                .types
                .get_span(ty)
                .to_range()
                .ok_or_else(|| fail("immediate structure has no source span"))?;
            let parts = tokenize(source.get(span).ok_or_else(|| fail("invalid type span"))?)?;
            let type_source = source
                .get(
                    module
                        .types
                        .get_span(ty)
                        .to_range()
                        .ok_or_else(|| fail("immediate structure has no source span"))?,
                )
                .ok_or_else(|| fail("invalid type span"))?;
            let mut i = 0;
            while i + 2 < parts.len() {
                if parts[i].text == "@" && parts[i + 1].text == "align" && parts[i + 2].text == "("
                {
                    let mut depth = 1usize;
                    let mut end = i + 3;
                    while end < parts.len() && depth != 0 {
                        if parts[end].text == "(" {
                            depth += 1;
                        }
                        if parts[end].text == ")" {
                            depth -= 1;
                        }
                        end += 1;
                    }
                    if depth != 0 || end == i + 4 {
                        return Err(fail("invalid alignment expression"));
                    }
                    let expr = &type_source[parts[i + 2].end..parts[end - 1].start];
                    alignment = alignment.max(evaluate_alignment(source, expr)?);
                    i = end;
                } else {
                    i += 1;
                }
            }
            for member in members {
                alignment = alignment.max(type_alignment(module, member.ty, source, layouts)?);
            }
        }
        naga::TypeInner::Array { base, .. } => {
            alignment = alignment.max(type_alignment(module, *base, source, layouts)?)
        }
        _ => {}
    }
    Ok(alignment)
}

// Naga evaluates @align constant expressions while lowering structures, then
// discards their alignment metadata. A synthetic one-member struct exposes the
// evaluated alignment as its size. It is appended only to the parser input.
fn evaluate_alignment(source: &str, expr: &str) -> Result<u32> {
    let tokens = tokenize(source)?;
    let declarations = declarations(&tokens)?;
    let name = (0u32..)
        .map(|i| format!("_fluxel_align_probe_{i}"))
        .find(|name| !tokens.iter().any(|t| t.text == name))
        .ok_or_else(|| fail("cannot generate alignment probe"))?;
    let input = format!("{source}\nstruct {name} {{ @align(max(4u, u32({expr}))) x: u32, }}\n");
    let module = parse(&input, &tokens, &declarations)?;
    let ty = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some(&name))
        .map(|(_, ty)| ty)
        .ok_or_else(|| fail("alignment probe missing"))?;
    match ty.inner {
        naga::TypeInner::Struct { span, .. } if span.is_power_of_two() => Ok(span),
        _ => Err(fail("invalid alignment probe")),
    }
}

fn has_immediate_pointer(tokens: &[Token<'_>]) -> bool {
    tokens.windows(3).any(|w| {
        w[0].text == "ptr" && w[1].text == "<" && matches!(w[2].text, "immediate" | "push_constant")
    })
}

fn contains_array(module: &naga::Module, ty: naga::Handle<naga::Type>) -> bool {
    match &module.types[ty].inner {
        naga::TypeInner::Array { .. } => true,
        naga::TypeInner::Struct { members, .. } => {
            members.iter().any(|m| contains_array(module, m.ty))
        }
        _ => false,
    }
}

fn flatten_arrays(
    source: &str,
    offset: u32,
    entry: &str,
    tokens: &[Token<'_>],
    declarations: &[(usize, usize)],
    module: &naga::Module,
) -> Result<String> {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| Error::Shader(format!("WebGPU immediate array validation: {e}")))?;
    if !module.entry_points.iter().any(|ep| ep.name == entry) {
        return Err(fail("pipeline entry not found"));
    }
    let mut layouts = naga::proc::Layouter::default();
    layouts
        .update(module.to_ctx())
        .map_err(|e| Error::Shader(format!("WebGPU immediate ABI layout: {e}")))?;
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    let mut globals = Vec::new();
    for (index, &(start, end)) in declarations.iter().enumerate() {
        let name = tokens[start + 4].text;
        let (handle, global) = module
            .global_variables
            .iter()
            .find(|(_, g)| g.name.as_deref() == Some(name))
            .ok_or_else(|| fail("global name resolution failed"))?;
        let word_count = exact_layout(module, global.ty, source, &layouts)?
            .0
            .checked_add(3)
            .ok_or_else(|| fail("immediate type too large"))?
            / 4;
        let unique = (0u32..)
            .map(|n| format!("_fluxel_immediate_flat_{index}_{n}"))
            .find(|name| !tokens.iter().any(|t| t.text == name))
            .ok_or_else(|| fail("cannot generate identifier"))?;
        let structure = format!("{unique}_type");
        if tokens.iter().any(|t| t.text == structure) {
            return Err(fail("generated type name collision"));
        }
        let mut flat = format!("struct {structure} {{ ");
        if offset != 0 {
            flat.push_str(&format!("@size({offset}) _pad: u32, "));
        }
        for word in 0..word_count {
            flat.push_str(&format!("w{word}: u32, "));
        }
        flat.push_str(&format!("}};\nvar<immediate> {unique}: {structure};\n"));
        edits.push((tokens[start].start, tokens[start].start, flat));
        edits.push((
            tokens[start + 2].start,
            tokens[start + 2].end,
            "private".into(),
        ));
        let mut init = String::new();
        flatten_type(
            module, global.ty, name, &unique, 0, &mut init, source, &layouts,
        )?;
        globals.push((handle, init));
        let _ = end;
    }
    for window in tokens.windows(3) {
        if window[0].text == "ptr"
            && window[1].text == "<"
            && matches!(window[2].text, "immediate" | "push_constant")
        {
            edits.push((window[2].start, window[2].end, "private".into()));
        }
    }
    for (ep_index, ep) in module.entry_points.iter().enumerate() {
        let active: Vec<_> = globals
            .iter()
            .filter(|(handle, _)| !info.get_entry_point(ep_index)[*handle].is_empty())
            .collect();
        if active.len() > 1 {
            return Err(fail("entry accesses multiple immediate variables"));
        }
        if let Some((_, initialization)) = active.first() {
            let pos = tokens
                .windows(2)
                .position(|w| w[0].text == "fn" && w[1].text == ep.name)
                .ok_or_else(|| fail("entry source unavailable"))?;
            let body = tokens[pos + 2..]
                .iter()
                .find(|t| t.text == "{")
                .ok_or_else(|| fail("entry body unavailable"))?;
            edits.push((body.end, body.end, format!("\n{initialization}\n")));
        }
    }
    edits.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let mut result = source.to_owned();
    for (start, end, replacement) in edits {
        result.replace_range(start..end, &replacement);
    }
    let required = tokens.iter().enumerate().any(|(i, t)| {
        t.text == "requires"
            && tokens[i + 1..]
                .iter()
                .take_while(|t| t.text != ";")
                .any(|t| t.text == "immediate_address_space")
    });
    if required {
        Ok(result)
    } else {
        Ok(format!("requires immediate_address_space;\n{result}"))
    }
}

fn flatten_type(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    target: &str,
    storage: &str,
    base: u32,
    out: &mut String,
    source: &str,
    layouts: &naga::proc::Layouter,
) -> Result<()> {
    use naga::TypeInner as T;
    match &module.types[ty].inner {
        T::Scalar(s) => flatten_scalar(*s, target, storage, base, out),
        T::Vector { size, scalar } => {
            for i in 0..*size as u32 {
                flatten_scalar(
                    *scalar,
                    &format!("{target}[{i}]"),
                    storage,
                    base + i * scalar.width as u32,
                    out,
                )?;
            }
            Ok(())
        }
        T::Matrix {
            columns,
            rows,
            scalar,
        } => {
            let row_count = *rows as u32;
            let alignment = if row_count == 2 { 2 } else { 4 } * scalar.width as u32;
            let stride = (row_count * scalar.width as u32 + alignment - 1) & !(alignment - 1);
            for col in 0..*columns as u32 {
                for row in 0..row_count {
                    flatten_scalar(
                        *scalar,
                        &format!("{target}[{col}][{row}]"),
                        storage,
                        base + col * stride + row * scalar.width as u32,
                        out,
                    )?;
                }
            }
            Ok(())
        }
        T::Array {
            base: element,
            size: naga::ArraySize::Constant(count),
            ..
        } => {
            let (size, align, _) = exact_layout(module, *element, source, layouts)?;
            let stride = round_up(size, align)?;
            for i in 0..count.get() {
                flatten_type(
                    module,
                    *element,
                    &format!("{target}[{i}]"),
                    storage,
                    base + i * stride,
                    out,
                    source,
                    layouts,
                )?;
            }
            Ok(())
        }
        T::Struct { members, .. } => {
            let (_, _, offsets) = exact_layout(module, ty, source, layouts)?;
            for (member, member_offset) in members.iter().zip(offsets) {
                let name = member
                    .name
                    .as_ref()
                    .ok_or_else(|| fail("unnamed structure member"))?;
                flatten_type(
                    module,
                    member.ty,
                    &format!("{target}.{name}"),
                    storage,
                    base + member_offset,
                    out,
                    source,
                    layouts,
                )?;
            }
            Ok(())
        }
        _ => Err(fail("immediate array contains unsupported type")),
    }
}

fn flatten_scalar(
    scalar: naga::Scalar,
    target: &str,
    storage: &str,
    offset: u32,
    out: &mut String,
) -> Result<()> {
    let word = format!("{storage}.w{}", offset / 4);
    let value = match (scalar.kind, scalar.width, offset % 4) {
        (naga::ScalarKind::Uint, 4, 0) => word,
        (naga::ScalarKind::Sint, 4, 0) => format!("bitcast<i32>({word})"),
        (naga::ScalarKind::Float, 4, 0) => format!("bitcast<f32>({word})"),
        (naga::ScalarKind::Float, 2, 0) => format!("bitcast<vec2<f16>>({word})[0]"),
        (naga::ScalarKind::Float, 2, 2) => format!("bitcast<vec2<f16>>({word})[1]"),
        _ => return Err(fail("immediate scalar layout unsupported")),
    };
    out.push_str(&format!("{target} = {value};\n"));
    Ok(())
}

fn round_up(value: u32, alignment: u32) -> Result<u32> {
    value
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or_else(|| fail("immediate layout overflow"))
}

// Reconstruct WGSL layout: Naga's layouter loses explicit nested structure
// alignment, so its offsets and array stride cannot be used in this case.
fn exact_layout(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    source: &str,
    layouts: &naga::proc::Layouter,
) -> Result<(u32, u32, Vec<u32>)> {
    match &module.types[ty].inner {
        naga::TypeInner::Array {
            base,
            size: naga::ArraySize::Constant(count),
            ..
        } => {
            let (size, align, _) = exact_layout(module, *base, source, layouts)?;
            let total = round_up(size, align)?
                .checked_mul(count.get())
                .ok_or_else(|| fail("immediate layout overflow"))?;
            Ok((total, align, Vec::new()))
        }
        naga::TypeInner::Struct { members, .. } => {
            let span = module
                .types
                .get_span(ty)
                .to_range()
                .ok_or_else(|| fail("structure source unavailable"))?;
            let text = source
                .get(span)
                .ok_or_else(|| fail("invalid structure source"))?;
            let parts = tokenize(text)?;
            let mut previous = 0usize;
            let mut total = 0u32;
            let mut struct_align = 1u32;
            let mut offsets = Vec::new();
            for member in members {
                let name = member
                    .name
                    .as_deref()
                    .ok_or_else(|| fail("unnamed structure member"))?;
                let index = (previous..parts.len().saturating_sub(1))
                    .find(|&i| parts[i].text == name && parts[i + 1].text == ":")
                    .ok_or_else(|| fail("structure member source unavailable"))?;
                let (member_size, mut member_align, _) =
                    exact_layout(module, member.ty, source, layouts)?;
                let mut actual_size = member_size;
                let mut i = previous;
                while i + 2 < index {
                    if parts[i].text == "@"
                        && matches!(parts[i + 1].text, "align" | "size")
                        && parts[i + 2].text == "("
                    {
                        let attribute = parts[i + 1].text;
                        let mut nesting = 1usize;
                        let mut end = i + 3;
                        while end < index && nesting != 0 {
                            if parts[end].text == "(" {
                                nesting += 1;
                            }
                            if parts[end].text == ")" {
                                nesting -= 1;
                            }
                            end += 1;
                        }
                        if nesting != 0 {
                            return Err(fail("invalid structure attribute"));
                        }
                        let expr = &text[parts[i + 2].end..parts[end - 1].start];
                        let value = evaluate_constant(source, expr)?;
                        if attribute == "align" {
                            if !value.is_power_of_two() || value < member_align {
                                return Err(fail("invalid structure alignment"));
                            }
                            member_align = value;
                        } else {
                            if value < member_size {
                                return Err(fail("invalid structure size"));
                            }
                            actual_size = value;
                        }
                        i = end;
                    } else {
                        i += 1;
                    }
                }
                total = round_up(total, member_align)?;
                offsets.push(total);
                total = total
                    .checked_add(actual_size)
                    .ok_or_else(|| fail("immediate layout overflow"))?;
                struct_align = struct_align.max(member_align);
                previous = index + 2;
            }
            Ok((round_up(total, struct_align)?, struct_align, offsets))
        }
        _ => Ok((
            layouts[ty].size,
            layouts[ty]
                .alignment
                .to_string()
                .parse()
                .map_err(|_| fail("invalid type alignment"))?,
            Vec::new(),
        )),
    }
}

fn evaluate_constant(source: &str, expr: &str) -> Result<u32> {
    let tokens = tokenize(source)?;
    let declarations = declarations(&tokens)?;
    let name = (0u32..)
        .map(|i| format!("_fluxel_constant_probe_{i}"))
        .find(|name| !tokens.iter().any(|t| t.text == name))
        .ok_or_else(|| fail("cannot generate constant probe"))?;
    let input = format!("{source}\nconst {name} = u32({expr});\n");
    let module = parse(&input, &tokens, &declarations)?;
    let c = module
        .constants
        .iter()
        .find(|(_, c)| c.name.as_deref() == Some(&name))
        .map(|(_, c)| c)
        .ok_or_else(|| fail("constant probe missing"))?;
    match module.global_expressions[c.init] {
        naga::Expression::Literal(naga::Literal::U32(value)) => Ok(value),
        _ => Err(fail("constant expression could not be evaluated")),
    }
}
