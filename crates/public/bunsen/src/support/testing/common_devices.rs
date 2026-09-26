/* TODO(burn 0.22 migration): burn::tensor::backend was removed, and burn::tensor::Device does not implement DeviceOps. Use &Device. See BREAKING_CHANGES.md #27, #35. */
use burn::{
    tensor::backend::DeviceOps,
};

/// Selected burn backend for fast-setup tests.
/* TODO(burn 0.22 migration): a backend type alias is a Device constructor, for example Device::flex() or Device::cuda(0). Replace each B: Backend parameter that uses this alias with a &Device argument. See BREAKING_CHANGES.md #27. */
pub type CpuBackend = ::burn::backend::Flex;

cfg_select! {
    feature = "cuda" => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = ::burn::backend::Cuda;
    }
    feature = "metal" => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = ::burn::backend::Metal;
    }
    feature = "vulkan" => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = ::burn::backend::Vulkan;
    }
    feature = "wgpu" => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = ::burn::backend::Wgpu;
    }
    _ => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = CpuBackend;
    }
}
