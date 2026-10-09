//! Narrow hardware-evidence surface for the standalone GL-family fixtures.
//!
//! It returns reports only; neither a native context nor a portable device can
//! escape.  The platform adapters own WGL/EGL creation and currentness.

/// A minimum desktop OpenGL core version requested by the Windows evidence
/// fixture.
///
/// This is a fixture input, not a portable RHI capability level. A value is
/// accepted only for the v13 desktop support interval, GL 4.0 through GL 4.6.
/// Omitting it from the fixture keeps the production-like "highest available"
/// WGL selection path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DesktopGlVersion {
    major: u8,
    minor: u8,
}

impl DesktopGlVersion {
    /// Creates a supported desktop GL minimum-version test request.
    pub const fn new(major: u8, minor: u8) -> Option<Self> {
        if major == 4 && minor <= 6 {
            Some(Self { major, minor })
        } else {
            None
        }
    }

    /// The requested major component.
    pub const fn major(self) -> u8 {
        self.major
    }

    /// The requested minor component.
    pub const fn minor(self) -> u8 {
        self.minor
    }
}

/// Raw colour data read from the fixture's own colour target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColourReadback {
    pub extent: [u32; 2],
    pub bytes: Vec<u8>,
    pub row_order: &'static str,
}

/// A real context's driver-discovery projection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGlContextReport {
    /// The optional minimum version requested by the fixture.
    ///
    /// `None` means WGL's ordinary highest-available fallback was used. When
    /// present, [`Self::version`] is guaranteed to be this version or newer,
    /// rather than necessarily equal to it.
    pub requested_minimum_version: Option<String>,
    pub profile: String,
    /// The version string observed from the actual current driver context.
    pub version: String,
    pub shading_language_version: String,
    pub vendor: String,
    pub renderer: String,
    pub driver_or_browser: String,
    pub debug: bool,
    pub forward_compatible: bool,
    pub robust_access: bool,
    pub no_error: bool,
    pub other_flags: Vec<String>,
    pub reported_extension_count: usize,
    pub reported_extensions: Vec<String>,
    pub typed_extensions: Vec<(String, String)>,
    pub capabilities: Vec<(String, bool)>,
    pub limits: Vec<(String, String)>,
    pub surface_facts: String,
    pub drawable_extent: [u32; 2],
    pub owner_thread: String,
}

/// Per-state-domain traffic observed during one fixture execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DomainTally {
    pub domain: String,
    pub requests: u64,
    pub emitted: u64,
    pub skipped: u64,
    pub unknown_recoveries: u64,
}

/// Measured native draw execution and optional readback evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGlDrawReport {
    pub context: NativeGlContextReport,
    pub mode: String,
    pub draws_requested: u32,
    pub passes: u64,
    pub pass_loads: u64,
    pub pass_stores: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_created: u64,
    pub cache_evicted: u64,
    pub cache_live_entries: u64,
    pub cache_live_bytes: u64,
    pub steady_state_allocations: u64,
    pub binding_bytes_copied: u64,
    pub domains: Vec<DomainTally>,
    pub submit_nanos: u64,
    pub total_nanos: u64,
    pub drawable_extent: [u32; 2],
    pub colour: Option<ColourReadback>,
}

fn validate_request(
    extent: [u32; 2],
    identity: u64,
    mode: &str,
    draws: Option<u32>,
) -> Result<(), String> {
    if identity == 0 {
        return Err("a device identity has to be nonzero".into());
    }
    if extent.contains(&0) {
        return Err("a native drawable extent has to be nonzero".into());
    }
    if !matches!(mode, "optimized" | "oracle") {
        return Err(format!(
            "unknown execution mode `{mode}`; expected `optimized` or `oracle`"
        ));
    }
    if draws == Some(0) {
        return Err("the evidence workload requires at least one draw".into());
    }
    Ok(())
}
