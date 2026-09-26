//! # Compat Tensor Operations

use burn::{
    Tensor,
    };

/// Maps nan and infinities to numbers.
pub fn nan_to_num<const D: usize>(
    tensor: Tensor<D>,
    nan_val: f64,
    neg_inf_val: f64,
    pos_inf_val: f64,
) -> Tensor<D> {
    let is_nan = tensor.clone().is_nan();
    let is_inf = tensor.clone().is_inf();
    let is_neg = tensor.clone().lower_elem(0.0);

    let pos_inf = is_inf.clone().bool_and(is_neg.clone().bool_not());
    let neg_inf = is_inf.clone().bool_and(is_neg);

    tensor
        .mask_fill(is_nan, nan_val)
        .mask_fill(neg_inf, neg_inf_val)
        .mask_fill(pos_inf, pos_inf_val)
}
