#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "../utils.h"

// CUDA methods
void SP_cuda(int blocks,
			 int threads,
			 int version,
			 long d,
			 long m,
			 long k,
			 long d_block_size,
			 long k_block_size,
			 torch::Tensor g,
			 torch::Tensor indices,
			 torch::Tensor values,
			 torch::Tensor out,
			 int use_bf16);

void LCG_cuda(int blocks,
              int threads,
              int version,
              long d,
              long m,
              long k,
              long d_block_size,
              long k_block_size,
              torch::Tensor c,
              torch::Tensor indices,
              torch::Tensor values,
              torch::Tensor out,
              int use_bf16);

// C++ methods callable from Python
void SP(int blocks,
		int threads,
		int version,
		long d,
		long m,
		long k,
		long d_block_size,
		long k_block_size,
		torch::Tensor g,
		torch::Tensor indices,
		torch::Tensor values,
		torch::Tensor out,
		int use_bf16) {
	CHECK_INPUT(g);
	CHECK_INPUT(indices);
	CHECK_INPUT(values);
	CHECK_INPUT(out);
	CHECK_THREADS(threads);

	const at::cuda::OptionalCUDAGuard device_guard(device_of(g));
	SP_cuda(blocks, threads, version, d, m, k, d_block_size, k_block_size, g, indices, values, out, use_bf16);
}

void LCG(int blocks,
         int threads,
         int version,
         long d,
         long m,
         long k,
         long d_block_size,
         long k_block_size,
         torch::Tensor c,
         torch::Tensor indices,
         torch::Tensor values,
         torch::Tensor out,
         int use_bf16) {
	CHECK_INPUT(c);
	CHECK_INPUT(indices);
	CHECK_INPUT(values);
	CHECK_INPUT(out);
	CHECK_THREADS(threads);

  	const at::cuda::OptionalCUDAGuard device_guard(device_of(c));
	LCG_cuda(blocks, threads, version, d, m, k, d_block_size, k_block_size, c, indices, values, out, use_bf16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("SP", &SP, "G x g dot products (CUDA)");
	m.def("LCG", &LCG, "Sum c_i * g_i (CUDA)");
}
