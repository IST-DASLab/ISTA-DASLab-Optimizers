#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

typedef long long LL;

// CUDA methods
int get_max_floats_for_shared_memory_per_thread_block_cuda();
int get_sm_count_cuda();
void zerorize_block_components_cuda(torch::Tensor vector,
                                    torch::Tensor indices,
                                    LL d,
                                    LL k,
                                    LL d_block_size,
                                    LL k_block_size);
void copy_values_large_to_small_cuda(LL d,
                                     LL k,
                                     LL d_block_size,
                                     LL k_block_size,
                                     torch::Tensor indices,
                                     torch::Tensor vector,
                                     torch::Tensor out);
void copy_values_small_to_large_cuda(LL d,
                                     LL k,
                                     LL d_block_size,
                                     LL k_block_size,
                                     torch::Tensor indices,
                                     torch::Tensor vector,
                                     torch::Tensor out);

// C++ methods
int get_max_floats_for_shared_memory_per_thread_block() {
	return get_max_floats_for_shared_memory_per_thread_block_cuda();
}

int get_sm_count() {
	return get_sm_count_cuda();
}

void zerorize_block_components_cuda(torch::Tensor vector,
                                    torch::Tensor indices,
                                    LL d,
                                    LL k,
                                    LL d_block_size,
                                    LL k_block_size) {
    CHECK_INPUT(vector);
    CHECK_INPUT(indices);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(vector));
    zerorize_block_components_cuda(vector, indices, d, k, d_block_size, k_block_size);
}

void void copy_values_large_to_small(LL d,
                                          LL k,
                                          LL d_block_size,
                                          LL k_block_size,
                                          torch::Tensor indices,
                                          torch::Tensor vector,
                                          torch::Tensor out) {
    CHECK_INPUT(indices);
    CHECK_INPUT(vector);
    CHECK_INPUT(out);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(indices));
    copy_values_large_to_small_cuda(d, k, d_block_size, k_block_size, indices, vector, out);
}

void copy_values_small_to_large(LL d,
                                     LL k,
                                     LL d_block_size,
                                     LL k_block_size,
                                     torch::Tensor indices,
                                     torch::Tensor vector,
                                     torch::Tensor out) {
    CHECK_INPUT(indices);
    CHECK_INPUT(vector);
    CHECK_INPUT(out);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(indices));
    copy_values_small_to_large_cuda(d, k, d_block_size, k_block_size, indices, vector, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("get_max_floats_for_shared_memory_per_thread_block", &get_max_floats_for_shared_memory_per_thread_block,
		  "Computes the maximum number of floats that can be stored in the shared memory per thread block");
	m.def("get_sm_count", &get_sm_count, "Return number of SMs for the GPU");
	m.def("zerorize_block_components", &zerorize_block_components, "Zerorizes the components in blocks.");
	m.def("copy_values_large_to_small", &copy_values_large_to_small, "Copy values from `vector` at `indices` to out.");
	m.def("copy_values_small_to_large", &copy_values_small_to_large, "Copy values from `vector` at `indices` to out.");
}
