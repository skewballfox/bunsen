use burn::{
    module::Module,
    };

use crate::errors::{
    BunsenResult,
    WithOkOrPanic,
};

/// Trait for initializing a Module.
pub trait ModuleInit<M: Module> {
    /// Initializes a Module.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<M>;

    /// Initializes a Module, panicking on error.
    fn init(
        &self,
        device: &Device,
    ) -> M {
        self.try_init(device).ok_or_panic()
    }
}
