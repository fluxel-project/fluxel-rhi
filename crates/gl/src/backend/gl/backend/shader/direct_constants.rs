//! Safe direct-GLSL pipeline-constant lowering.
//!
//! Direct GLSL has no portable specialization-constant object model.  Fluxel
//! therefore defines the deliberately small direct-source ABI: every pipeline
//! constant is emitted as a preprocessor definition whose name is the Core
//! constant name.  Definitions are placed immediately after `#version`, so
//! they are visible to extensions, declarations, and `main` alike.  Only
//! values which have a single, injection-free GLSL spelling are accepted.

use std::collections::BTreeSet;

use fluxel_rhi_core::api::{Error, Result, SpecializationValue};

use crate::backend::gl::api::{GlShaderDialect, GlShaderKey, GlShaderSource};

use super::source_key;

pub(super) fn apply(
    source: &mut GlShaderSource,
    constants: &[(String, SpecializationValue)],
) -> Result<()> {
    if constants.is_empty() {
        return Ok(());
    }

    let mut version_end = version_directive_end(&source.text).ok_or(Error::Unsupported(
        "direct GLSL pipeline constants require #version",
    ))?;
    let mut seen = BTreeSet::new();
    let mut definitions = String::new();
    for (name, value) in constants {
        validate_name(name)?;
        if !seen.insert(name.as_str()) {
            return Err(Error::Shader(format!(
                "duplicate direct GLSL pipeline constant {name}"
            )));
        }
        definitions.push_str("#define ");
        definitions.push_str(name);
        definitions.push(' ');
        definitions.push_str(&literal(source.dialect, *value)?);
        definitions.push('\n');
    }
    if !source.text[..version_end].ends_with('\n') {
        source.text.insert(version_end, '\n');
        version_end += 1;
    }
    source.text.insert_str(version_end, &definitions);
    source.source_hash = GlShaderKey(source_key(source.text.as_bytes(), &source.entry_point));
    Ok(())
}

fn version_directive_end(source: &str) -> Option<usize> {
    // GLSL permits whitespace and comments before its first directive. Scan
    // comments rather than matching their contents as executable directives.
    let mut remaining = source;
    loop {
        remaining = remaining.trim_start();
        if let Some(comment) = remaining.strip_prefix("//") {
            remaining = &comment[comment.find('\n')?..];
        } else if let Some(comment) = remaining.strip_prefix("/*") {
            remaining = &comment[comment.find("*/")? + 2..];
        } else {
            break;
        }
    }
    let suffix = remaining.strip_prefix("#version")?;
    if !suffix.chars().next().is_some_and(char::is_whitespace) {
        return None;
    }
    let start = source.len() - remaining.len();
    Some(start + remaining.find('\n').map_or(remaining.len(), |end| end + 1))
}

fn validate_name(name: &str) -> Result<()> {
    let mut characters = name.chars();
    let Some(first) = characters.next() else {
        return Err(Error::Shader(
            "direct GLSL pipeline constant name is empty".into(),
        ));
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || !characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
    {
        return Err(Error::Shader(format!(
            "direct GLSL pipeline constant {name:?} is not an identifier"
        )));
    }
    if name.starts_with("gl_")
        || name.starts_with("GL_")
        || name.starts_with("fluxel_")
        || name == "main"
    {
        return Err(Error::Shader(format!(
            "direct GLSL pipeline constant {name:?} uses a reserved identifier"
        )));
    }
    Ok(())
}

fn literal(dialect: GlShaderDialect, value: SpecializationValue) -> Result<String> {
    let unsupported = || Error::Unsupported("direct GLSL pipeline constant value");
    match value {
        SpecializationValue::Bool(value) => Ok(value.to_string()),
        SpecializationValue::U8(value) => Ok(format!("{value}u")),
        SpecializationValue::U16(value) => Ok(format!("{value}u")),
        SpecializationValue::U32(value) => Ok(format!("{value}u")),
        SpecializationValue::I8(value) => Ok(value.to_string()),
        SpecializationValue::I16(value) => Ok(value.to_string()),
        SpecializationValue::I32(value) => Ok(value.to_string()),
        // GLSL's ordinary scalar vocabulary has no portable 64-bit integer
        // literal; accepting one would silently change its type.
        SpecializationValue::U64(_) | SpecializationValue::I64(_) => Err(unsupported()),
        SpecializationValue::F16Bits(bits) => float_literal(f16_to_f32(bits)),
        SpecializationValue::F32(value) => float_literal(value),
        SpecializationValue::F64(value) => match dialect {
            GlShaderDialect::Desktop { version } if version >= 400 => double_literal(value),
            GlShaderDialect::Desktop { .. } | GlShaderDialect::Embedded { .. } => {
                Err(unsupported())
            }
        },
    }
}

fn float_literal(value: f32) -> Result<String> {
    if !value.is_finite() {
        return Err(Error::Unsupported(
            "non-finite direct GLSL pipeline constant",
        ));
    }
    let mut literal = value.to_string();
    if !literal.contains(['.', 'e', 'E']) {
        literal.push_str(".0");
    }
    Ok(literal)
}

fn double_literal(value: f64) -> Result<String> {
    if !value.is_finite() {
        return Err(Error::Unsupported(
            "non-finite direct GLSL pipeline constant",
        ));
    }
    let mut literal = value.to_string();
    if !literal.contains(['.', 'e', 'E']) {
        literal.push_str(".0");
    }
    literal.push_str("LF");
    Ok(literal)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut fraction = fraction;
            let mut exponent = 127 - 14;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((fraction & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        exponent => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::gl::api::{GlShaderDialect, GlShaderStage};

    fn source(text: &str) -> GlShaderSource {
        GlShaderSource {
            stage: GlShaderStage::Fragment,
            dialect: GlShaderDialect::Desktop { version: 460 },
            entry_point: "main".into(),
            source_hash: GlShaderKey([0; 32]),
            text: text.into(),
            spirv: None,
            debug_name: None,
        }
    }

    #[test]
    fn constants_are_inserted_after_version_with_stable_literals() {
        let mut source = source(
            "#version 460 core\n#extension GL_ARB_gpu_shader_fp64 : require\nvoid main() {}\n",
        );
        apply(
            &mut source,
            &[
                ("COUNT".into(), SpecializationValue::U32(4)),
                ("scale".into(), SpecializationValue::F32(1.0)),
                ("toggle".into(), SpecializationValue::Bool(true)),
            ],
        )
        .unwrap();
        assert!(source.text.starts_with(
            "#version 460 core\n#define COUNT 4u\n#define scale 1.0\n#define toggle true\n"
        ));
        assert!(source.text.contains("#extension GL_ARB_gpu_shader_fp64"));
    }

    #[test]
    fn constants_reject_injection_reserved_and_unrepresentable_values() {
        let mut source = source("#version 460 core\nvoid main() {}\n");
        assert!(
            apply(
                &mut source,
                &[("bad\n#define injected".into(), SpecializationValue::I32(1))]
            )
            .is_err()
        );
        assert!(
            apply(
                &mut source,
                &[("gl_Position".into(), SpecializationValue::I32(1))]
            )
            .is_err()
        );
        assert!(apply(&mut source, &[("wide".into(), SpecializationValue::U64(1))]).is_err());
        assert!(
            apply(
                &mut source,
                &[("nan".into(), SpecializationValue::F32(f32::NAN))]
            )
            .is_err()
        );
    }

    #[test]
    fn version_scanner_skips_license_comments_and_fake_directives() {
        let mut source =
            source("/* license\n#version 100\n*/ // header\n#version 460 core\nvoid main() {}\n");
        apply(
            &mut source,
            &[("GAIN".into(), SpecializationValue::F32(0.5))],
        )
        .unwrap();
        assert!(
            source
                .text
                .contains("#version 460 core\n#define GAIN 0.5\n")
        );
        assert!(version_directive_end("/* unclosed #version 460\n").is_none());
    }

    #[test]
    fn unterminated_version_line_is_separated_from_definitions() {
        let mut source = source("#version 460 core");
        apply(
            &mut source,
            &[("GAIN".into(), SpecializationValue::F32(0.5))],
        )
        .unwrap();
        assert_eq!(source.text, "#version 460 core\n#define GAIN 0.5\n");
    }
}
