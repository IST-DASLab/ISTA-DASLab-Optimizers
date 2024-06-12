#ifndef __UTILS_H__
#define __UTILS_H__

#include <torch/all.h>
#include <torch/python.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <assert.h>
#include <vector>
#include <cstring>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <limits> // for epsilon

using namespace std;

typedef __nv_bfloat16 bfloat16;
typedef __nv_bfloat162 bfloat162;
typedef short int int16;
typedef long long LL;

inline void gpuAssert(cudaError_t code, const char *file, int line) {
	/*
		https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
		https://ori-cohen.medium.com/https-medium-com-real-life-cuda-programming-part-0-an-overview-f83a2cd77779
		https://stackoverflow.com/tags/cuda/info
		https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
		https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism
		https://docs.nvidia.com/cuda/thrust/index.html
	*/
    if(code != cudaSuccess) {
        fprintf(stderr, "Error detected by GPU Assert:\n\tError %d: %s \n\tFile: %s\n\tLine: %d\n",
            (int) code, cudaGetErrorString(code), file, line);
        exit(666);
    }
}

__host__ __device__ inline long div_inc(long a, long b) {
	long r = a / b;
	return (a % b > 0) ? (r + 1) : r;
}

__device__ inline LL minLL(LL a, LL b) {
   return (a < b) ? a : b;
}

__host__ __device__ inline float my_pow(float base, int exp) { // log2 time
   float result = 1.;
   while(exp > 0) {
       if(exp & 1) {
           result *= base;
           --exp;
       }
       base = base * base;
       exp >>= 1;
   }
   return result;
}

__device__ inline long log_threads(long T) {
	if(T == 2) return 1;
	if(T == 4) return 2;
	if(T == 8) return 3;
	if(T == 16) return 4;
	if(T == 32) return 5;
	if(T == 64) return 6;
	if(T == 128) return 7;
	if(T == 256) return 8;
	if(T == 512) return 9;
	if(T == 1024) return 10;
}

inline LL get_threads(LL max_threads) {
    /*
        This method computes the number of threads as the first power of 2 larger than or equal to max_threads
    */
    LL threads = 1;
    while(threads < max_threads) {
        threads <<= 1;
    }
    return threads;
}

__device__ inline void dynamically_assign(void *out, void *inp, int out_index, int inp_index, int out_bits, int inp_bits) {
    /*
        This function assigns out[out_index] = inp[inp_index] based on the types and performs the conversions when needed:
        - when out and inp are both BFloat16 or Float, the assignment is done directly without any conversion
        - when out and inp have different data types, the correct functions (float-to-bfloat16 or bfloat16-to-float) are called accordingly
        We do not have implicit cast between float and bfloat16, we have to call specific CUDA functions for this
    */
    if(out_bits == inp_bits) { // out and inp have the same type, so assign without any conversion
        if(out_bits == 16) { // both are bfloat16
            ((bfloat16*)out)[out_index] = ((bfloat16*)inp)[inp_index];
        } else { // both are float
            ((float*)out)[out_index] = ((float*)inp)[inp_index];
        }
    } else { // out and inp have different types, we have to check both to know which conversion function we have to call
        if(out_bits == 16 && inp_bits == 32) { // convert float to bfloat16
            ((bfloat16*)out)[out_index] = __float2bfloat16(((float*)inp)[inp_index]);
        } else if (out_bits == 32 && inp_bits == 16) { // convert bfloat16 to float
            ((float*)out)[out_index] = __bfloat162float(((bfloat16*)inp)[inp_index]);
        }
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_THREADS(T) assert(T == 2 || T == 32 || T == 64 || T == 128 || T == 256 || T == 512 || T == 1024);

#define FLOAT_EPS std::numeric_limits<float>::epsilon()
#define DOUBLE_EPS std::numeric_limits<double>::epsilon()
#define GPU_ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define ASSERT_BF16(x) { assert(torch::ScalarType::BFloat16 == x.scalar_type()); }
#define ASSERT_FLOAT_16_OR_32(x) { assert(torch::ScalarType::BFloat16 == x.scalar_type() || torch::ScalarType::Float == x.scalar_type()); }

#define COPY_DIRECTION_k2d 0
#define COPY_DIRECTION_d2k 1

#endif