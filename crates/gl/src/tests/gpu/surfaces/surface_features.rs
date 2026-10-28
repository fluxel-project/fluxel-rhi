//! WebGL2 surface regressions grouped by observable surface behavior.
//!
//! Each child owns one GPU scenario and its pixel oracle. This module only
//! assembles the fixture tree so target-specific test registration stays local.

use super::*;

#[path = "surface_features/depth_publish.rs"]
mod depth_publish;
#[path = "surface_features/integer_clear.rs"]
mod integer_clear;
#[path = "surface_features/mrt_load.rs"]
mod mrt_load;
#[path = "surface_features/msaa_resolve_linear.rs"]
mod msaa_resolve_linear;
#[path = "surface_features/msaa_resolve_srgb.rs"]
mod msaa_resolve_srgb;
#[path = "surface_features/orientation.rs"]
mod orientation;
