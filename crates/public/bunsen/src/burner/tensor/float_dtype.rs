use burn::{
    tensor::{
        DType,
        Element,
    },
};

/// The backend's default float [`DType`]: `B::FloatElem`'s.
///
/// This is what a `Tensor<B, D>` built without a dtype of its own gets —
/// `Tensor::random`, `Tensor::zeros`, `Tensor::from_data` — so it is the
/// dtype an operation between a model's output and a freshly-built tensor
/// needs both sides to be in.
///
/// A module whose parameters were loaded at some other precision computes in
/// *that* precision; this is the dtype its **interface** speaks. Casting a
/// result to it at the module boundary lets the caller stay in the backend's
/// float without knowing what the checkpoint shipped as.
pub fn backend_float_dtype<B: Backend>() -> DType {
    </* TODO(burn 0.22 migration): B::FloatElem no longer exists -- the backend's float dtype is now a runtime Device setting (device.settings().float_dtype), not a compile-time associated type. This call site needs a &Device argument threaded in; see BREAKING_CHANGES.md #14. */ B::FloatElem as Element>::dtype()
}

#[cfg(test)]
mod tests {
    use burn::tensor::DType;

    use super::*;
    use crate::support::testing::{
        CpuBackend,
        default_device,
    };

    /// The helper agrees with what an undtyped tensor actually gets: that
    /// equality is the whole point of it.
    #[test]
    fn test_matches_a_default_tensor() {
        type B = CpuBackend;
        let device = default_device();

        let t: burn::Tensor<1> = burn::Tensor::zeros([2], &device);
        assert_eq!(backend_float_dtype::<B>(), t.dtype());
        assert_eq!(backend_float_dtype::<B>(), DType::F32);
    }
}
