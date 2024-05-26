#include "utils.h"

__global__ void asymm_global_quant_kernel(LL d, uint8_t *xq, float x_min, float x_max, float *x) {
	/*
		This kernel computes xq = Q(x, x_min, x_max) for 4 bits (implements point 4 from PhD notebook page 55)
	*/
	float2 *x2 = reinterpret_cast<float2*>(x); // we will read two values from x at once

	const LL B = gridDim.x; // number of blocks
	const LL Bid = blockIdx.x; // block id
// 	const LL T = blockDim.x; // number of threads
	const LL Tid = threadIdx.x; // thread id

	LL index = Bid * B + Tid; // this is the index used for x2 and xq
	LL d2 = d >> 1; // d2 will be the size of xq

	float u = (x_max - x_min) / 15.0f; // 15 = 16 - 1 = 2^4 - 1

	uint8_t msb; // the MSB of a xq component
	uint8_t lsb; // the LSB of a xq component
	float2 vx2; // the value that will store x2[index]

	if(index < d2) {
		vx2 = x2[index];
		msb = (uint8_t) floorf((vx2.x - x_min) / u + 0.5f);
		lsb = (uint8_t) floorf((vx2.y - x_min) / u + 0.5f);

		xq[index] = (msb << 4) | lsb;
	}

	if((d & 1) && (Tid == 0) && (Bid == 0)) { // if d is odd, this means that
        float vx = x[d - 1];
        msb = (uint8_t) floorf((vx - x_min) / u + 0.5f);
        xq[d2] = (msb << 4);
	}
}
void asymm_global_quant_cuda(int blocks, int threads, LL d, torch::Tensor xq, float x_min, float x_max, torch::Tensor x) {
	asymm_global_quant_kernel<<<blocks, threads>>>(
		d,
		(uint8_t*) xq.data_ptr(),
		x_min,
		x_max,
		(float*)x.data_ptr());
	// error checks
	gpuErrorCheck(cudaGetLastError());
	gpuErrorCheck(cudaPeekAtLastError());
// 	gpuErrorCheck(cudaDeviceSynchronize());
}
