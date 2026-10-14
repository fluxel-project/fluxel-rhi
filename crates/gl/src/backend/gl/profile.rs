//! Canonical profile and version facts for one GL-family context.
//!
//! Discovery normalizes platform strings into these values before capability
//! routing starts.  They contain neither a native context handle nor browser
//! state, so desktop GL, GLES, and WebGL2 can share the route registry.

/// A GL core-version requirement or observed native version.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GlVersion {
    pub(crate) major: u8,
    pub(crate) minor: u8,
}

impl GlVersion {
    pub(crate) const fn new(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    pub(crate) const fn is_met_by(self, actual: Self) -> bool {
        actual.major > self.major || (actual.major == self.major && actual.minor >= self.minor)
    }
}

/// The API family selected for a context generation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlProfile {
    Desktop { major: u8, minor: u8 },
    Embedded { major: u8, minor: u8 },
    WebGl2,
}

impl GlProfile {
    pub(crate) const DESKTOP_MINIMUM: GlVersion = GlVersion::new(4, 0);
    pub(crate) const GLES_MINIMUM: GlVersion = GlVersion::new(3, 0);

    /// Returns whether this profile meets Fluxel's supported GL floor.
    pub(crate) const fn accepts(self) -> bool {
        match self {
            Self::Desktop { major, minor } => {
                Self::DESKTOP_MINIMUM.is_met_by(GlVersion::new(major, minor))
            }
            Self::Embedded { major, minor } => {
                Self::GLES_MINIMUM.is_met_by(GlVersion::new(major, minor))
            }
            Self::WebGl2 => true,
        }
    }

    /// Returns whether an ordinary core feature route exists in this profile.
    pub(crate) const fn has_core(self, desktop: GlVersion, gles: GlVersion) -> bool {
        match self {
            Self::Desktop { major, minor } => desktop.is_met_by(GlVersion::new(major, minor)),
            Self::Embedded { major, minor } => gles.is_met_by(GlVersion::new(major, minor)),
            Self::WebGl2 => false,
        }
    }

    /// Historical feature routes express independent desktop and GLES floors.
    /// Keeping that query here lets route selection stay profile driven while
    /// the public Core capability vocabulary remains platform-neutral.
    pub(crate) const fn meets(self, desktop: Option<GlVersion>, gles: Option<GlVersion>) -> bool {
        match self {
            Self::Desktop { major, minor } => match desktop {
                Some(required) => required.is_met_by(GlVersion::new(major, minor)),
                None => false,
            },
            Self::Embedded { major, minor } => match gles {
                Some(required) => required.is_met_by(GlVersion::new(major, minor)),
                None => false,
            },
            Self::WebGl2 => false,
        }
    }
}
