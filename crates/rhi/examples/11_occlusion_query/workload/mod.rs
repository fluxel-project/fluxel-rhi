//! Occlusion-query workload modules for resource setup, recording, and readback.

use super::*;

#[path = "completion.rs"]
mod completion;
#[path = "creation.rs"]
mod creation;
#[path = "recording.rs"]
mod recording;
