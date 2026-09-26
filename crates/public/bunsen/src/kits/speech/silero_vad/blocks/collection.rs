use burn::{
    module::Module,
    };

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        WithOkOrPanic,
    },
    kits::speech::silero_vad::SileroVad,
};

/// Collection of sample-rate Silero VAD models.
#[derive(Module, Debug)]
pub struct SileroVadCollection {
    /// Per-sample-rate models.
    pub branches: Vec<(usize, SileroVad)>,
}

impl SileroVadCollection {
    /// Look up the metadata for the given sample rate.
    ///
    /// # Arguments
    /// * `sample_rate`: Sample rate in Hz.
    pub fn try_branch(
        &self,
        sample_rate: usize,
    ) -> BunsenResult<&SileroVad> {
        self.branches
            .iter()
            .find(|(rate, _)| *rate == sample_rate)
            .map(|(_, vad)| vad)
            .ok_or_else(|| {
                BunsenError::ResourceNotFound(format!(
                    "sample_rate {sample_rate} not found in {:?}",
                    self.branches
                        .iter()
                        .map(|(rate, _)| *rate)
                        .collect::<Vec<_>>()
                ))
            })
    }

    /// Select the branch for the given signal rate.
    ///
    /// Panics if the sample rate is not supported.
    pub fn expect_branch(
        &self,
        sample_rate: usize,
    ) -> &SileroVad {
        self.try_branch(sample_rate).ok_or_panic()
    }
}
