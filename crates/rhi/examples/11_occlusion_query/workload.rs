//! Occlusion-query workload modules for resource setup, recording, and readback.

use super::*;

#[path = "workload/completion.rs"]
mod completion;
#[path = "workload/creation.rs"]
mod creation;
#[path = "workload/recording.rs"]
mod recording;
