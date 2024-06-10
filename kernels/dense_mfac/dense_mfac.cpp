#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

torch::Tensor hinv_setup_cuda(torch::Tensor tmp, torch::Tensor coef);
torch::Tensor hinv_mul_cuda(int rows, torch::Tensor giHig, torch::Tensor giHix);

torch::Tensor hinv_setup(torch::Tensor tmp, torch::Tensor coef) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(tmp));
  return hinv_setup_cuda(tmp, coef);
}

torch::Tensor hinv_mul(int rows, torch::Tensor giHig, torch::Tensor giHix) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(giHig));
  return hinv_mul_cuda(rows, giHig, giHix);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hinv_setup", &hinv_setup, "Hinv setup (CUDA)");
  m.def("hinv_mul", &hinv_mul, "Hinv mul (CUDA)");
}
