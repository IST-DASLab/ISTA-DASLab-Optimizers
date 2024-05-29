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
				  torch::Tensor g,
				  torch::Tensor indices,
				  torch::Tensor values,
				  torch::Tensor out,
				  int use_bf16,
				  long d_block_size,
				  long k_block_size);

void LCG_cuda(int blocks,
				   int threads,
				   int version,
				   long d,
				   long m,
				   long k,
				   torch::Tensor c,
				   torch::Tensor indices,
				   torch::Tensor values,
				   torch::Tensor out,
				   int use_bf16,
				   long d_block_size,
				   long k_block_size);

int info_cuda(int numThreads, int sharedMemSize, bool verbose);

int get_max_floats_for_shared_memory_per_thread_block_cuda();

int get_sm_count_cuda();

// C++ methods callable from Python
void SP(int blocks, int threads, int version, long d, long m, long k, torch::Tensor g, torch::Tensor indices, torch::Tensor values, torch::Tensor out, int use_bf16, long d_block_size, long k_block_size) {
	CHECK_INPUT(g);
	CHECK_INPUT(indices);
	CHECK_INPUT(values);
	CHECK_INPUT(out);
	CHECK_THREADS(threads);

	const at::cuda::OptionalCUDAGuard device_guard(device_of(g));
	SP_cuda(blocks, threads, version, d, m, k, g, indices, values, out, use_bf16, d_block_size, k_block_size);
}

void LCG(int blocks, int threads, int version, long d, long m, long k, torch::Tensor c, torch::Tensor indices, torch::Tensor values, torch::Tensor out, int use_bf16, long d_block_size, long k_block_size) {
//	CHECK_INPUT(mem);
	CHECK_INPUT(c);
	CHECK_INPUT(indices);
	CHECK_INPUT(values);
	CHECK_INPUT(out);
	CHECK_THREADS(threads);

  	const at::cuda::OptionalCUDAGuard device_guard(device_of(c));
	LCG_cuda(blocks, threads, version, d, m, k, c, indices, values, out, use_bf16, d_block_size, k_block_size);
}

int info(int numThreads, int sharedMemSize, bool verbose=false) {
	return info_cuda(numThreads, sharedMemSize, verbose);
}

int get_max_floats_for_shared_memory_per_thread_block() {
	return get_max_floats_for_shared_memory_per_thread_block_cuda();
}

int get_sm_count() {
	return get_sm_count_cuda();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("SP", &SP, "G x g dot products (CUDA)");
	m.def("LCG", &LCG, "Sum c_i * g_i (CUDA)");
	m.def("info", &info, "Get maximum number of blocks to maximize GPU occupancy");
	m.def("get_max_floats_for_shared_memory_per_thread_block", &get_max_floats_for_shared_memory_per_thread_block,
		  "Computes the maximum number of floats that can be stored in the shared memory per thread block");
	m.def("get_sm_count", &get_sm_count, "Return number of SMs for the GPU");
}
