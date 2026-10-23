//! Current-context entry points for multisample array storage.

use crate::backend::gl::api::{GlExtensionSet, GlFamilyProfile};
use core::ffi::c_void;

pub(super) type StorageFn = unsafe extern "system" fn(u32, i32, u32, i32, i32, i32, u8);

#[derive(Clone, Copy, Default)]
pub(super) struct MultisampleArrayFns {
    image: Option<StorageFn>,
    storage: Option<StorageFn>,
    oes: Option<StorageFn>,
}

impl MultisampleArrayFns {
    /// Loads symbols only; the profile/extension route is checked at use.
    pub(super) unsafe fn load(loader: impl Fn(&str) -> *const c_void) -> Self {
        let load_one = |name| {
            let address = loader(name);
            // SAFETY: GL defines these three entry points with the same ABI;
            // the integer-sized internal-format signedness has identical ABI.
            (!address.is_null())
                .then(|| unsafe { core::mem::transmute::<*const c_void, StorageFn>(address) })
        };
        Self {
            image: load_one("glTexImage3DMultisample"),
            storage: load_one("glTexStorage3DMultisample"),
            oes: load_one("glTexStorage3DMultisampleOES"),
        }
    }

    pub(super) fn select(
        self,
        profile: GlFamilyProfile,
        extensions: &GlExtensionSet,
    ) -> Option<StorageFn> {
        match profile {
            GlFamilyProfile::Desktop { major, minor } if (major, minor) >= (4, 3) => {
                self.storage.or(self.image)
            }
            GlFamilyProfile::Desktop { .. } => self.image,
            GlFamilyProfile::Embedded { major, minor } if (major, minor) >= (3, 2) => self.storage,
            GlFamilyProfile::Embedded { major, minor }
                if (major, minor) >= (3, 1)
                    && extensions
                        .raw_reported_names()
                        .any(|name| name == "GL_OES_texture_storage_multisample_2d_array") =>
            {
                self.oes
            }
            _ => None,
        }
    }
}
