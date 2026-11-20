//! Converts source compiler feedback to portable diagnostics.

use fluxel_rhi_core::api::{
    Error, Result, ShaderCompilationReport, ShaderDiagnostic, ShaderDiagnosticLocation,
    ShaderDiagnosticSeverity, ShaderDiagnosticSource,
};

pub(super) fn parse_wgsl(source: &str) -> Result<naga::Module> {
    naga::front::wgsl::parse_str(source).map_err(|error| {
        wgsl_report(
            source,
            error.message().into(),
            error.labels(),
            error.emit_to_string(source),
        )
    })
}

pub(super) fn wgsl_validation(
    source: &str,
    error: naga::WithSpan<naga::valid::ValidationError>,
) -> Error {
    wgsl_report(
        source,
        error.to_string(),
        error.spans().map(|(span, label)| (*span, label.as_str())),
        error.emit_to_string(source),
    )
}

fn wgsl_report<'a>(
    source: &str,
    message: String,
    labels: impl Iterator<Item = (naga::Span, &'a str)>,
    rendered: String,
) -> Error {
    let notes: Vec<_> = labels
        .map(|(span, label)| ShaderDiagnostic {
            severity: ShaderDiagnosticSeverity::Note,
            message: label.into(),
            location: wgsl_location(source, span),
        })
        .collect();
    let mut diagnostics = vec![ShaderDiagnostic {
        severity: ShaderDiagnosticSeverity::Error,
        message,
        location: notes.iter().find_map(|note| note.location.clone()),
    }];
    diagnostics.extend(notes);
    Error::ShaderCompilation(ShaderCompilationReport {
        source: ShaderDiagnosticSource::Wgsl,
        diagnostics,
        rendered,
    })
}

fn wgsl_location(source: &str, span: naga::Span) -> Option<ShaderDiagnosticLocation> {
    let range = span.to_range()?;
    source.get(range.clone())?;
    let location = span.location(source);
    Some(ShaderDiagnosticLocation {
        file: None,
        line: location.line_number,
        column: Some(location.line_position),
        byte_range: Some(range),
    })
}

pub(super) fn dxc_report(
    source: ShaderDiagnosticSource,
    rendered: String,
) -> ShaderCompilationReport {
    let mut diagnostics: Vec<_> = rendered.lines().filter_map(dxc_diagnostic).collect();
    if !diagnostics
        .iter()
        .any(|note| note.severity == ShaderDiagnosticSeverity::Error)
    {
        diagnostics.insert(
            0,
            ShaderDiagnostic {
                severity: ShaderDiagnosticSeverity::Error,
                message: rendered.trim().into(),
                location: None,
            },
        );
    }
    ShaderCompilationReport {
        source,
        diagnostics,
        rendered,
    }
}

pub(super) fn dxc_feedback(
    source: ShaderDiagnosticSource,
    rendered: String,
) -> ShaderCompilationReport {
    let diagnostics = rendered.lines().filter_map(dxc_diagnostic).collect();
    ShaderCompilationReport {
        source,
        diagnostics,
        rendered,
    }
}

fn dxc_diagnostic(line: &str) -> Option<ShaderDiagnostic> {
    use ShaderDiagnosticSeverity::*;
    for (marker, severity) in [
        ("fatal error:", Error),
        ("error:", Error),
        ("warning:", Warning),
        ("note:", Note),
    ] {
        let Some((prefix, message)) = line.split_once(marker) else {
            continue;
        };
        let prefix = prefix.trim();
        let location = if prefix.is_empty() || prefix == "dxc:" {
            None
        } else {
            let Some(location) = dxc_location(prefix) else {
                continue;
            };
            Some(location)
        };
        return Some(ShaderDiagnostic {
            severity,
            message: message.trim().into(),
            location,
        });
    }
    None
}

fn dxc_location(prefix: &str) -> Option<ShaderDiagnosticLocation> {
    let (file_and_line, tail) = prefix.strip_suffix(':')?.rsplit_once(':')?;
    let last = tail.parse::<u32>().ok().filter(|&value| value != 0)?;
    let (file, line, column) = match file_and_line.rsplit_once(':') {
        Some((file, line)) if line.parse::<u32>().is_ok() => {
            (file, line.parse::<u32>().ok()?, Some(last))
        }
        _ => (file_and_line, last, None),
    };
    if line == 0 || file.is_empty() {
        return None;
    }
    Some(ShaderDiagnosticLocation {
        file: Some(file.into()),
        line,
        column,
        byte_range: None,
    })
}
