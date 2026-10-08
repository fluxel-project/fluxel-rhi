//! Apple platform adapters shared by the macOS and iOS example runners.
//!
//! `fluxel-host` owns the native application lifecycle and surface callbacks;
//! this module composes those callbacks with the portable RHI presentation API.

#![cfg(target_vendor = "apple")]
