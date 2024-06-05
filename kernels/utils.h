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
using namespace cooperative_groups;

#define FLOAT_EPS std::numeric_limits<float>::epsilon()
#define DOUBLE_EPS std::numeric_limits<double>::epsilon()
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

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

__device__ long div_inc(long a, long b) {
	long r = a / b;
	return (a % b > 0) ? (r + 1) : r;
}

//#define ALIGN_BYTES 8
//typedef __nv_bfloat16 bfloat16;

// 2 bytes = 16 bits for each bfloat16 => 8 bytes = 64 bits in total for the entire struct
// aligning to 8 bytes means that we can read the entire struct in one 64 bit transaction
//typedef struct __align__(ALIGN_BYTES) { bfloat16 x, y, z, w; } bfloat164; // chapter 10.27.6.1 from the CUDA Bible
//typedef struct __align__(ALIGN_BYTES) { bfloat16 x, y; } bfloat162;
//typedef struct __align__(ALIGN_BYTES) { float x, y; } float2;

#endif