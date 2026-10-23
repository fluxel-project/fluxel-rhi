//! Fixed-function stencil-plane transfers assembled from WebGL2 primitives.
//!
//! WebGL2 cannot upload or read a standalone stencil plane directly.  The
//! write path stages bytes in a private integer texture and applies the eight
//! stencil bits with ordinary raster draws.  It remains an immediate native
//! lowering: no CPU readback, command cache, render graph, or hazard tracker
//! is introduced here.

mod read;
mod write;
