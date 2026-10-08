//! Browser platform adapters shared by the WebGPU and WebGL2 example runners.
//!
//! The browser owns canvas, animation-frame, visibility, resize, and context/device
//! loss events through `fluxel-jsbridge`. RHI resource and submission ownership
//! stays in the backend; browser-native objects remain private to this module.

#![cfg(target_arch = "wasm32")]
