#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include "../utils.h"

typedef long long LL;

// CUDA methods
void compute_microadam_update_cuda(int blocks, int threads, int carveout,
						  	   LL t, float beta1, float beta2, float eps,
						  	   LL d_block_size, LL k_block_size,
						       LL d, LL m, LL k,
						       torch::Tensor indices, torch::Tensor values, torch::Tensor out);

void asymm_block_quant_cuda(LL d, LL q_block_size, torch::Tensor xq, torch::Tensor xmin, torch::Tensor xmax, torch::Tensor x);
void asymm_block_quant_inv_cuda(LL d, LL q_block_size, torch::Tensor xq, torch::Tensor xmin, torch::Tensor xmax, torch::Tensor x, float alpha);

// C++ methods
void compute_microadam_update(int blocks, int threads, int carveout,
						  LL t, float beta1, float beta2, float eps,
						  LL d_block_size, LL k_block_size,
						  LL d, LL m, LL k,
						  torch::Tensor indices, torch::Tensor values, torch::Tensor out) {
	CHECK_INPUT(indices);
	CHECK_INPUT(values);
	CHECK_INPUT(out);
	CHECK_THREADS(threads);

  	const at::cuda::OptionalCUDAGuard device_guard(device_of(indices));
	compute_microadam_update_cuda(blocks, threads, carveout,
							  t, beta1, beta2, eps,
							  d_block_size, k_block_size,
							  d, m, k,
							  indices, values, out);
}

void asymm_block_quant(LL d, LL q_block_size, torch::Tensor xq, torch::Tensor xmin, torch::Tensor xmax, torch::Tensor x) {
	CHECK_INPUT(xq);
	CHECK_INPUT(xmin);
	CHECK_INPUT(xmax);
	CHECK_INPUT(x);

	const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
	asymm_block_quant_cuda(d, q_block_size, xq, xmin, xmax, x);
}

void asymm_block_quant_inv(LL d, LL q_block_size, torch::Tensor xq, torch::Tensor xmin, torch::Tensor xmax, torch::Tensor x, float alpha) {
	CHECK_INPUT(xq);
	CHECK_INPUT(xmin);
	CHECK_INPUT(xmax);
	CHECK_INPUT(x);

	const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
	asymm_block_quant_inv_cuda(d, q_block_size, xq, xmin, xmax, x, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("compute_microadam_update", &compute_microadam_update, "Computes the update from Compressed Adam.");

    m.def("asymm_block_quant", &asymm_block_quant, "Asymmetrically quantizes a vector to 4bits in blocks.");
    m.def("asymm_block_quant_inv", &asymm_block_quant_inv, "Asymmetrically dequantizes a vector to 4bits in blocks.");
}
