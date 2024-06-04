/* These methods will be called from Python */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_THREADS(T) assert(T == 2 || T == 32 || T == 64 || T == 128 || T == 256 || T == 512 || T == 1024);

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
	SP_cuda(blocks, threads, version, m, k, d_block_size, k_block_size, g, indices, values, out, use_bf16);
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
	LCG_cuda(blocks, threads, version, d_block_size, k_block_size, d, m, k, c, indices, values, out, use_bf16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("SP", &SP, "G x g dot products (CUDA)");
	m.def("LCG", &LCG, "Sum c_i * g_i (CUDA)");
}
