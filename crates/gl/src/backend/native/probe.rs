//! Version parsing shared by WGL and EGL discovery.

use crate::backend::api::GlFamilyProfile;

/// Why a native GL version string cannot enter the accepted profile set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NativeProfileError {
    Empty,
    Malformed(String),
    BelowDesktopFloor { major: u8, minor: u8 },
    BelowEmbeddedFloor { major: u8, minor: u8 },
}

/// Parses the exact GL_VERSION spelling without treating a desktop string as
/// ES just because a vendor name happens to contain "ES".
pub(crate) fn parse_native_profile(version: &str) -> Result<GlFamilyProfile, NativeProfileError> {
    let value = version.trim();
    if value.is_empty() {
        return Err(NativeProfileError::Empty);
    }
    let (embedded, number) = if let Some(number) = value.strip_prefix("OpenGL ES ") {
        (true, number)
    } else {
        (false, value)
    };
    let number = number.split_whitespace().next().unwrap_or_default();
    let mut parts = number.split('.');
    let parse = |part: Option<&str>| part.and_then(|value| value.parse::<u8>().ok());
    let (Some(major), Some(minor)) = (parse(parts.next()), parse(parts.next())) else {
        return Err(NativeProfileError::Malformed(value.into()));
    };
    if embedded {
        if major < 3 {
            Err(NativeProfileError::BelowEmbeddedFloor { major, minor })
        } else {
            Ok(GlFamilyProfile::Embedded { major, minor })
        }
    } else if major < 4 {
        Err(NativeProfileError::BelowDesktopFloor { major, minor })
    } else {
        Ok(GlFamilyProfile::Desktop { major, minor })
    }
}

#[cfg(test)]
#[path = "../../tests/backend/native/probe__tests.rs"]
mod tests;
