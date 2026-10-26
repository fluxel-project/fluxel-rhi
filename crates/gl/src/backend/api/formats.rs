//! Exact format facts for one GL-family context.
//!
//! Facts are either unconditional guarantees of the selected API profile or
//! the result of a concrete operation probe.  A guarantee is not described as
//! a probe: providers may add operation-probed facts later without changing
//! the meaning of the baseline record.

use std::collections::{BTreeMap, BTreeSet};

use super::{GlExtensionSet, GlFamilyProfile, GlKnownExtension, GlLimits};

#[path = "format_types.rs"]
mod format_types;
pub(crate) use format_types::*;

impl GlFormat {
    /// Whether the current implementation can use this compressed format on
    /// TEXTURE_3D. Exact format discovery is still required independently.
    /// BPTC specifies a stack of independently encoded 2D slices; S3TC,
    /// RGTC and ETC2/EAC do not permit this target in desktop core GL.
    pub(crate) fn supports_compressed_3d(
        self,
        profile: GlFamilyProfile,
        extensions: &GlExtensionSet,
    ) -> bool {
        if !matches!(profile, GlFamilyProfile::Desktop { .. }) {
            return false;
        }
        let hdr = extensions.is_acquired(GlKnownExtension::KhrTextureCompressionAstcHdr);
        match self {
            Self::Bc6hRgbUfloat | Self::Bc6hRgbSfloat | Self::Bc7RgbaUnorm | Self::Bc7RgbaSrgb => {
                true
            }
            Self::Astc {
                color_space: GlCompressedColorSpace::Hdr,
                ..
            } => hdr,
            Self::Astc {
                color_space: GlCompressedColorSpace::Linear | GlCompressedColorSpace::Srgb,
                ..
            } => {
                hdr || (extensions.is_acquired(GlKnownExtension::KhrTextureCompressionAstcLdr)
                    && extensions.is_acquired(GlKnownExtension::KhrTextureCompressionAstcSliced3d))
            }
            _ => false,
        }
    }
}

/// The resource class to which an exact format fact applies.
///
/// Multisample renderbuffers and multisample textures have different GL
/// contracts.  In particular, WebGL2 exposes the former but not the latter.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlFormatResourceKind {
    /// A texture, including a single-sample texture when `sample_count` is 1.
    Texture,
    /// A two-dimensional multisample array texture.
    MultisampleArray,
    /// A renderbuffer attachment.
    Renderbuffer,
}

/// Origin of an exact format fact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlFormatEvidence {
    /// An unconditional guarantee of the selected GL4, GLES3, or WebGL2 API.
    CoreGuaranteed,
    /// An exact, callable compressed-texture extension on this same context.
    ExtensionAcquired(GlKnownExtension),
    /// A context-specific operation was executed and succeeded.
    OperationProbed,
}

/// Exact use facts for one `(format, resource kind, sample_count)` tuple.
///
/// A false field is a conservative unsupported fact, not an invitation for a
/// provider to emulate the operation. Storage facts stay split because
/// read/write support is not implied by each other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlFormatCapabilities {
    /// Queried format.
    pub format: GlFormat,
    /// Texture or renderbuffer contract to which this fact applies.
    pub resource_kind: GlFormatResourceKind,
    /// Exact sample count. Zero is invalid and rejected by the table.
    pub sample_count: u32,
    /// Whether this is a profile guarantee or concrete operation evidence.
    pub evidence: GlFormatEvidence,
    /// Sampling from this format/count is legal.
    pub sampled: bool,
    /// Linear filtering is legal for this format/count.
    pub filterable: bool,
    /// Framebuffer rendering is legal for this format/count.
    pub renderable: bool,
    /// Blending into a color attachment is legal for this format/count.
    pub blendable: bool,
    /// Storage image reads are legal for this format/count.
    pub storage_read: bool,
    /// Storage image writes are legal for this format/count.
    pub storage_write: bool,
    /// Copying from this format/count is legal.
    pub copy_source: bool,
    /// Copying to this format/count is legal.
    pub copy_destination: bool,
}

/// Immutable table keyed by exact format and sample count.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GlFormatTable {
    entries: BTreeMap<(GlFormat, GlFormatResourceKind, u32), GlFormatCapabilities>,
    exact_buffer_readback: BTreeSet<GlFormat>,
    exact_alias_readback: BTreeSet<GlFormat>,
    compressed_alias_readback: BTreeMap<GlFormat, GlFormat>,
}

impl GlFormatTable {
    /// Record exact byte-preserving pixel-pack support separately from blits.
    pub(crate) fn mark_exact_buffer_readback(&mut self, format: GlFormat) {
        self.exact_buffer_readback.insert(format);
    }

    pub(crate) fn exact_buffer_readback(&self, format: GlFormat) -> bool {
        self.exact_buffer_readback.contains(&format)
    }
    pub(crate) fn mark_exact_alias_readback(&mut self, format: GlFormat) {
        self.exact_alias_readback.insert(format);
    }

    pub(crate) fn exact_alias_readback(&self, format: GlFormat) -> bool {
        self.exact_alias_readback.contains(&format)
    }
    pub(crate) fn mark_compressed_alias_readback(&mut self, format: GlFormat, alias: GlFormat) {
        self.compressed_alias_readback.insert(format, alias);
    }

    pub(crate) fn compressed_alias_readback(&self, format: GlFormat) -> Option<GlFormat> {
        self.compressed_alias_readback.get(&format).copied()
    }
    /// Inserts one exact observation; conflicting repeated observations are rejected.
    pub(crate) fn record(
        &mut self,
        capabilities: GlFormatCapabilities,
    ) -> Result<(), GlFormatTableError> {
        if capabilities.sample_count == 0 {
            return Err(GlFormatTableError::ZeroSampleCount {
                format: capabilities.format,
            });
        }
        if capabilities.format.compressed_info().is_some()
            && (capabilities.resource_kind != GlFormatResourceKind::Texture
                || capabilities.sample_count != 1
                || capabilities.renderable
                || capabilities.blendable
                || capabilities.storage_read
                || capabilities.storage_write)
        {
            return Err(GlFormatTableError::InvalidCompressedUsage {
                format: capabilities.format,
            });
        }
        if (capabilities.storage_read || capabilities.storage_write)
            && capabilities.evidence != GlFormatEvidence::OperationProbed
        {
            return Err(GlFormatTableError::StorageRequiresOperationProbe {
                format: capabilities.format,
                resource_kind: capabilities.resource_kind,
                sample_count: capabilities.sample_count,
            });
        }
        let key = (
            capabilities.format,
            capabilities.resource_kind,
            capabilities.sample_count,
        );
        if let Some(previous) = self.entries.get(&key) {
            if previous != &capabilities {
                return Err(GlFormatTableError::ConflictingObservation {
                    format: capabilities.format,
                    sample_count: capabilities.sample_count,
                });
            }
            return Ok(());
        }
        self.entries.insert(key, capabilities);
        Ok(())
    }

    /// Returns exact texture facts; texture is the common resource contract.
    pub(crate) fn get(&self, format: GlFormat, sample_count: u32) -> Option<GlFormatCapabilities> {
        self.get_for(GlFormatResourceKind::Texture, format, sample_count)
    }

    /// Returns exact facts for a resource kind, format, and sample count.
    pub(crate) fn get_for(
        &self,
        resource_kind: GlFormatResourceKind,
        format: GlFormat,
        sample_count: u32,
    ) -> Option<GlFormatCapabilities> {
        self.entries
            .get(&(format, resource_kind, sample_count))
            .copied()
    }

    /// Iterates in stable `(format, resource kind, sample count)` order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = GlFormatCapabilities> + '_ {
        self.entries.values().copied()
    }

    /// Returns whether at least one exact format/count pair proves read/write storage.
    pub(crate) fn has_storage_read_write(&self) -> bool {
        self.entries.values().any(|facts| {
            facts.resource_kind == GlFormatResourceKind::Texture
                && facts.evidence == GlFormatEvidence::OperationProbed
                && facts.storage_read
                && facts.storage_write
        })
    }

    /// Verifies evidence against the one profile and extension ledger that
    /// will own this immutable format table.
    pub(crate) fn validate_evidence(
        &self,
        profile: GlFamilyProfile,
        extensions: &GlExtensionSet,
    ) -> Result<(), GlFormatTableError> {
        for facts in self.entries.values() {
            match facts.evidence {
                GlFormatEvidence::CoreGuaranteed => {
                    if !(facts.format.is_non_compressed_core_baseline_for(profile)
                        || facts.format.is_core_compressed_for(profile))
                    {
                        return Err(GlFormatTableError::InvalidCoreGuarantee {
                            format: facts.format,
                            profile,
                        });
                    }
                }
                GlFormatEvidence::ExtensionAcquired(extension) => {
                    if !extension.is_legal_for(profile) {
                        return Err(GlFormatTableError::ExtensionIllegalForProfile {
                            format: facts.format,
                            extension,
                            profile,
                        });
                    }
                    if !extensions.is_acquired(extension) {
                        return Err(GlFormatTableError::ExtensionNotAcquired {
                            format: facts.format,
                            extension,
                        });
                    }
                    if !facts.format.is_supplied_by_extension(extension) {
                        return Err(GlFormatTableError::ExtensionDoesNotSupplyFormat {
                            format: facts.format,
                            extension,
                        });
                    }
                }
                GlFormatEvidence::OperationProbed => {}
            }
        }
        Ok(())
    }

    /// Validates mandatory common format records and every observed sample count.
    pub(crate) fn validate_for_limits(&self, limits: &GlLimits) -> Result<(), GlFormatTableError> {
        for format in [
            GlFormat::Rgba8Unorm,
            GlFormat::Rgba8Srgb,
            GlFormat::Depth32Float,
        ] {
            if !self
                .entries
                .contains_key(&(format, GlFormatResourceKind::Texture, 1))
            {
                return Err(GlFormatTableError::MissingRequiredSampleOne { format });
            }
        }
        for facts in self.entries.values() {
            if facts.sample_count == 1 {
                continue;
            }
            let maximum = match facts.resource_kind {
                GlFormatResourceKind::Renderbuffer => limits.max_samples,
                GlFormatResourceKind::Texture | GlFormatResourceKind::MultisampleArray => {
                    match facts.format {
                        GlFormat::Depth16Unorm
                        | GlFormat::Depth24Unorm
                        | GlFormat::Depth24PlusStencil8
                        | GlFormat::Depth32Float
                        | GlFormat::Depth32FloatStencil8
                        | GlFormat::Stencil8 => limits.max_depth_texture_samples,
                        GlFormat::R8Uint
                        | GlFormat::R8Sint
                        | GlFormat::Rg8Uint
                        | GlFormat::Rg8Sint
                        | GlFormat::Rgba8Uint
                        | GlFormat::Rgba8Sint
                        | GlFormat::R16Uint
                        | GlFormat::R16Sint
                        | GlFormat::Rg16Uint
                        | GlFormat::Rg16Sint
                        | GlFormat::Rgba16Uint
                        | GlFormat::Rgba16Sint
                        | GlFormat::R32Uint
                        | GlFormat::R32Sint
                        | GlFormat::Rg32Uint
                        | GlFormat::Rg32Sint
                        | GlFormat::Rgba32Uint
                        | GlFormat::Rgba32Sint
                        | GlFormat::Rgb10a2Uint => limits.max_integer_samples,
                        _ => limits.max_color_texture_samples,
                    }
                }
            };
            if facts.sample_count > maximum {
                return Err(GlFormatTableError::SampleCountExceedsLimit {
                    format: facts.format,
                    sample_count: facts.sample_count,
                    maximum,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/backend/api/formats_tests.rs"]
mod tests;
/// Why a format fact cannot enter the durable table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlFormatTableError {
    /// Sample zero is not an actual GL sample count.
    ZeroSampleCount {
        /// Format with the invalid observation.
        format: GlFormat,
    },
    /// A compressed texture claimed an operation that this core contract forbids.
    InvalidCompressedUsage {
        /// Compressed format with an invalid resource or capability fact.
        format: GlFormat,
    },
    /// A static core claim is not guaranteed by this selected profile.
    InvalidCoreGuarantee {
        /// Format with an invalid static guarantee.
        format: GlFormat,
        /// Profile that does not guarantee it.
        profile: GlFamilyProfile,
    },
    /// The claimed extension is unavailable for the snapshot's profile.
    ExtensionIllegalForProfile {
        /// Format supported by the invalid extension claim.
        format: GlFormat,
        /// Extension from the format evidence.
        extension: GlKnownExtension,
        /// Snapshot profile.
        profile: GlFamilyProfile,
    },
    /// The bound extension ledger did not acquire the claimed extension.
    ExtensionNotAcquired {
        /// Format requiring the extension.
        format: GlFormat,
        /// Extension missing from the acquired ledger.
        extension: GlKnownExtension,
    },
    /// An acquired extension was used to claim an unrelated exact format.
    ExtensionDoesNotSupplyFormat {
        /// Claimed format.
        format: GlFormat,
        /// Acquired extension that cannot supply this format.
        extension: GlKnownExtension,
    },
    /// Two probes disagreed about the same exact pair.
    ConflictingObservation {
        /// Format with conflicting observations.
        format: GlFormat,
        /// Sample count with conflicting observations.
        sample_count: u32,
    },
    /// Storage image access is valid only after a concrete operation probe.
    StorageRequiresOperationProbe {
        /// Format with storage access facts.
        format: GlFormat,
        /// Resource class with storage access facts.
        resource_kind: GlFormatResourceKind,
        /// Exact sample count with storage access facts.
        sample_count: u32,
    },
    /// A common RHI format has no required single-sample fact.
    MissingRequiredSampleOne {
        /// Missing format.
        format: GlFormat,
    },
    /// An observed format/count pair exceeds the corresponding queried limit.
    SampleCountExceedsLimit {
        /// Format with an invalid count.
        format: GlFormat,
        /// Observed sample count.
        sample_count: u32,
        /// Applicable queried maximum.
        maximum: u32,
    },
}
