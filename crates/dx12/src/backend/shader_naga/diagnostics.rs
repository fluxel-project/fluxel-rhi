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
