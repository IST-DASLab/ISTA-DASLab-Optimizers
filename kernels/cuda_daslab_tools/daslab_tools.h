#ifndef __DASLAB_TOOLS_H__
#define __DASLAB_TOOLS_H__

#include "../utils.h"

// CUDA methods
inline int get_max_floats_for_shared_memory_per_thread_block_cuda();
inline int get_sm_count_cuda();
inline void zerorize_block_components_cuda(torch::Tensor vector,
                                           torch::Tensor indices,
                                           LL d,
                                           LL k,
                                           LL d_block_size,
                                           LL k_block_size);
inline void copy_values_large_to_small_cuda(LL d,
                                            LL k,
                                            LL d_block_size,
                                            LL k_block_size,
                                            torch::Tensor indices,
                                            torch::Tensor vector,
                                            torch::Tensor out);
inline void copy_values_small_to_large_cuda(LL d,
                                            LL k,
                                            LL d_block_size,
                                            LL k_block_size,
                                            torch::Tensor indices,
                                            torch::Tensor vector,
                                            torch::Tensor out);
__global__ inline void TestKernel(int n, int *v);
__global__ inline void zerorize_block_components_kernel_bf16(bfloat16 *vector,
                                                             int16 *indices,
                                                             LL d,
                                                             LL k,
                                                             LL d_block_size,
                                                             LL k_block_size);
__global__ inline void copy_values_large_to_small_kernel_bf16(LL d,
                                                              LL k,
                                                              LL d_block_size,
                                                              LL k_block_size,
                                                              int16 *indices,
                                                              bfloat16 *vector,
                                                              bfloat16 *out);
__global__ inline void copy_values_small_to_large_kernel_bf16(LL d,
                                                              LL k,
                                                              LL d_block_size,
                                                              LL k_block_size,
                                                              int16 *indices,
                                                              bfloat16 *vector,
                                                              bfloat16 *out);
#endif