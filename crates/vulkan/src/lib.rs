//! Vulkan lowering for Fluxel's statically typed execution API.
//!
//! The implementation keeps Vulkan ownership and command recording private to
//! this crate.  The public entry points select Vulkan once and return the
//! concrete Core types, exactly like the DX12 backend.

mod backend;

pub use backend::{
    Vulkan, VulkanExternalHandle, VulkanExternalMemoryHandle, VulkanExternalSync, VulkanInstance,
};
