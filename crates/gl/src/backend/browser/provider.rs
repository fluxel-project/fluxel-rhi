//! Context ownership and query provider for the WebGL2 discovery record.

use super::super::api::{GlContextLifecycle, GlError, GlFamilyApi, OwnerThreadIdentity};
use super::discovery::WebGl2BrowserDiscovery;

impl GlFamilyApi for WebGl2BrowserDiscovery {
    fn lifecycle(&self) -> GlContextLifecycle {
        self.lifecycle.get()
    }

    fn owner_thread(&self) -> OwnerThreadIdentity {
        self.owner_thread
    }

    fn assert_owner_thread(&self, operation: &'static str) -> Result<(), GlError> {
        WebGl2BrowserDiscovery::assert_owner_thread(self, operation)
    }

    fn discovery(&self) -> &super::super::api::GlDiscoverySnapshot {
        &self.snapshot
    }

    fn context_lost(&mut self) -> Result<(), GlError> {
        self.assert_owner_thread("context-lost")?;
        self.lifecycle.set(GlContextLifecycle::Lost);
        // Context loss invalidates query handles without calling methods on
        // dead JS objects. The BrowserOwner owns all other WebGL resources.
        self.clear_executable_state();
        Ok(())
    }
}
