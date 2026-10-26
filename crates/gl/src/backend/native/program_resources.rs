//! Native linked-program storage-block enumeration.
//!
//! Entry points belong to the provider's current GL context. Names and indices
//! describe the linked program and do not escape the native boundary.
use super::provider::NativeGlProvider;
use crate::backend::api::GlError;

type InterfaceFn = unsafe extern "system" fn(u32, u32, u32, *mut i32);
type NameFn = unsafe extern "system" fn(u32, u32, u32, i32, *mut i32, *mut i8);
#[derive(Clone, Copy)]
pub(super) struct StorageBlockEntries {
    interface: InterfaceFn,
    name: NameFn,
}
impl StorageBlockEntries {
    /// The loader resolves symbols for the provider's current context.
    pub(super) unsafe fn load(loader: &impl Fn(&str) -> *const core::ffi::c_void) -> Option<Self> {
        let interface = loader("glGetProgramInterfaceiv");
        let name = loader("glGetProgramResourceName");
        if interface.is_null() || name.is_null() {
            return None;
        }
        // SAFETY: the platform loader supplies these exact GL entry points.
        Some(Self {
            interface: unsafe {
                core::mem::transmute::<*const core::ffi::c_void, InterfaceFn>(interface)
            },
            name: unsafe { core::mem::transmute::<*const core::ffi::c_void, NameFn>(name) },
        })
    }
}
impl NativeGlProvider {
    pub(super) fn active_storage_blocks(
        &mut self,
        op: &'static str,
        program: glow::NativeProgram,
        numeric: bool,
    ) -> Result<Vec<(u32, String)>, GlError> {
        let entries = self.storage_block_entries.ok_or(GlError::Unsupported {
            operation: op,
            reason: "program storage-block enumeration unavailable",
        })?;
        let mut count = -1;
        let mut capacity = -1;
        // SAFETY: program is a live linked program in the current context; outputs are valid.
        unsafe {
            (entries.interface)(
                program.0.get(),
                glow::SHADER_STORAGE_BLOCK,
                glow::ACTIVE_RESOURCES,
                &mut count,
            );
            (entries.interface)(
                program.0.get(),
                glow::SHADER_STORAGE_BLOCK,
                glow::MAX_NAME_LENGTH,
                &mut capacity,
            );
        }
        self.driver_error(op)?;
        if count < 0 || (count > 0 && capacity <= 0) {
            return Err(Self::validation(
                op,
                "invalid program storage block metadata",
            ));
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut result = Vec::new();
        let mut buf = vec![0u8; capacity as usize];
        for index in 0..count as u32 {
            let mut len = -1;
            // SAFETY: buffer capacity matches the driver's reported maximum including terminator.
            unsafe {
                (entries.name)(
                    program.0.get(),
                    glow::SHADER_STORAGE_BLOCK,
                    index,
                    capacity,
                    &mut len,
                    buf.as_mut_ptr().cast(),
                );
            }
            self.driver_error(op)?;
            if len < 0 || len >= capacity {
                return Err(Self::validation(op, "invalid storage block name length"));
            }
            let name = std::str::from_utf8(&buf[..len as usize])
                .map_err(|_| Self::validation(op, "invalid storage block name encoding"))?;
            if numeric || name.is_empty() {
                let query = self.program_resource.ok_or_else(|| {
                    Self::validation(op, "program-interface binding query unavailable")
                })?;
                let (mut written, mut binding) = (0, -1);
                // SAFETY: `index` is a live storage-block resource index.
                unsafe {
                    query(
                        program.0.get(),
                        glow::SHADER_STORAGE_BLOCK,
                        index,
                        1,
                        [glow::BUFFER_BINDING].as_ptr(),
                        1,
                        &mut written,
                        &mut binding,
                    )
                };
                self.driver_error(op)?;
                if written != 1 || binding < 0 {
                    return Err(Self::validation(
                        op,
                        "unnamed storage block has no numeric binding",
                    ));
                }
                result.push((index, format!("@{binding}")));
            } else {
                result.push((index, name.to_owned()));
            }
        }
        Ok(result)
    }
}
