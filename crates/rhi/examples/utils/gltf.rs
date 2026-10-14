//! Small, deliberately constrained glTF 2.0 loader for the example assets.
//!
//! It accepts JSON `.gltf` files whose buffers are embedded base64 data URIs.
//! Output vertices have the `vkglTF::Vertex` ABI used by the source examples.

use std::{error::Error, fmt};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::Value;

#[path = "gltf/accessor.rs"]
mod accessor;
#[path = "gltf/json.rs"]
mod json;
#[path = "gltf/loader.rs"]
mod loader;
#[path = "gltf/math.rs"]
mod math;
#[path = "gltf/media.rs"]
mod media;
#[path = "gltf/types.rs"]
mod types;

pub use loader::{load_embedded_first_mesh, load_embedded_model};
pub use types::{
    DrawPrimitive, EmbeddedImage, EmbeddedMaterial, EmbeddedModel, EmbeddedSampler,
    EmbeddedTexture, GltfError, LoadOptions, VKGLTF_VERTEX_STRIDE,
};

use accessor::*;
use json::*;
use math::*;
use media::*;
use types::*;
