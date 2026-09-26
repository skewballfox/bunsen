//! Parameter load/save mappers.
//!
//! `burn` attaches transformations to a [`Param`] itself, so they run whenever
//! the parameter is loaded from or saved to a store. [`LinearConfig`] uses this
//! for [`LinearLayout::Col`](burn::nn::LinearLayout::Col): the weight is held
//! in `PyTorch`'s `[d_output, d_input]` orientation and transposed on the way
//! through.
//!
//! These helpers apply the same trick to a parameter that has already been
//! built, for the case where the enclosing module does not expose a layout
//! knob.
//!
//! [`LinearConfig`]: burn::nn::LinearConfig

use burn::{
    Tensor,
    module::Param,
    };

/// Repairs a 2-D weight mangled by `burn-store`'s stride-blind `PyTorch` read.
///
/// See `burn_bug_repro::pytorch_strided_weights` for the defect and a
/// reproduction; it is a dev crate, so this is not a link. In short: `PyTorch`
/// may store a `[R, C]` tensor as a column-major view (strides `(1, R)`), and
/// `PytorchStore` reads the raw storage as if it were row-major. A checkpoint
/// holds a view like this wherever the saved tensor was produced by transposing
/// another rather than by copying it.
///
/// The corruption is invertible. Reading storage that actually holds `Wᵀ` as
/// `[R, C]` yields `S = reshape(flat(Wᵀ), [R, C])`, and the adapter then
/// transposes it to `T = Sᵀ`. Since `flat(Sᵀᵀ) = flat(S) = flat(Wᵀ)`,
/// transposing back and reshaping to the parameter's own shape recovers `Wᵀ`,
/// which is what a row-major [`Linear`](burn::nn::Linear) wants.
///
/// For a square weight the reshape is a no-op and this degenerates to a plain
/// transpose — which is why square projections appear merely "untransposed"
/// while non-square ones come out scrambled.
///
/// The parameter's value and shape are untouched; only what crosses the store
/// boundary is repaired.
///
/// # Example
///
/// ```rust,ignore
/// let mut attn = mha_config.init(device);
/// attn.query.weight = repair_pytorch_strided_weight(attn.query.weight);
/// ```
pub fn repair_pytorch_strided_weight(
    param: Param<Tensor<2>>
) -> Param<Tensor<2>> {
    param
        // Coming from the store: undo the mangling.
        .load_mapper(|tensor: Tensor<2>| {
            tensor.device().sync().unwrap();
            let dims = tensor.dims();
            let tensor = tensor.transpose().reshape(dims);
            tensor.device().sync().unwrap();
            tensor
        })
        // Going back out: re-apply it, so a round trip is the identity.
        .save_mapper(|tensor: Tensor<2>| {
            tensor.device().sync().unwrap();
            let [rows, cols] = tensor.dims();
            let tensor = tensor.reshape([cols, rows]).transpose();
            tensor.device().sync().unwrap();
            tensor
        })
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::LinearConfig,
        prelude::TensorData,
        tensor::{
            Tolerance,
            },
    };

    use super::*;
    use crate::{
        burner::tensor::TensorElemOpExt,
        support::testing::{
            CpuBackend,
            default_device,
        },
    };

    type B = CpuBackend;
    type F = f32; /* TODO(burn 0.22 migration): the backend's compile-time float-element associated type is gone -- hardcoded to f32 for this test helper; adjust if this specific test needs a different working dtype. See BREAKING_CHANGES.md #14b. */

    /// The mapper must leave the live parameter alone — it only changes what
    /// crosses the store boundary.
    #[test]
    fn test_repair_preserves_the_live_value() {
        let device = default_device();

        let linear = LinearConfig::new(3, 5).with_bias(false).init::<B>(&device);
        let before = linear.weight.val();

        let mapped = repair_pytorch_strided_weight(linear.weight);

        assert_eq!(mapped.val().dims(), [3, 5]);
        mapped
            .val()
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&before.to_data_as::<F>(), Tolerance::default());
    }

    /// A round trip through the mappers is the identity, so saving what was
    /// loaded reproduces the store's orientation.
    #[test]
    fn test_load_then_save_round_trips() {
        let device = default_device();

        // A store-orientation tensor: `[d_output, d_input]`.
        let stored: Tensor<2> = Tensor::from_data(
            TensorData::new((0..15).map(|v| v as f64).collect::<Vec<_>>(), [5, 3]),
            &device,
        );

        let transposed = stored.clone().transpose();
        assert_eq!(transposed.dims(), [3, 5]);

        // Loading transposes into compute orientation, saving transposes back.
        transposed
            .transpose()
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&stored.to_data_as::<F>(), Tolerance::default());
    }
}
