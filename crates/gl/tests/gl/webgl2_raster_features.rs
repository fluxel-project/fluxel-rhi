//! Core WebGL2 MRT and dynamic-raster regression entrypoint.
//!
//! The shared fixtures retain pixel assertions for a hole at color output
//! location one. The paired Core fixture retains dynamic blend/stencil pixel
//! checks. Both issue through the public Core immediate encoder and the
//! BrowserOwner/BrowserEncoder production path.

use super::*;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) async fn run(device: &Device<Gl>) {
    mrt_hole(device).await;
}
