//! Type mapper for changing the data type of tensors in a module.
use burn::{
    Tensor,
    module::{
        ModuleMapper,
        Param,
    },
    tensor::DType,
};

/// Type mapper for changing the data type of tensors in a module.
pub struct DTypeMapper {
    /// Target data type for tensor conversion.
    pub dt: DType,

    /// Phantom data to ensure that `B` is a valid backend.
    phantom: std::marker::PhantomData<B>,
}

impl DTypeMapper {
    /// Creates a new type mapper with the specified target data type.
    pub fn new(dt: DType) -> Self {
        Self {
            dt,
            phantom: std::marker::PhantomData,
        }
    }
}

impl ModuleMapper<B> for DTypeMapper {
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<D>>,
    ) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        Param::from_mapped_value(id, tensor.cast(self.dt), mapper)
    }
}
