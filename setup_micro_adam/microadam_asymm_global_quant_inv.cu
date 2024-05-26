#include "utils.h"

__global__ void asymm_global_quant_inv_kernel(LL d, uint8_t *xq, float x_min, float x_max, float *x) {
	/*
		This kernel computes x += Q_inv(xq, x_min, x_max) for 4 bits (implements point 1 from PhD notebook page 55)
		Here, x is the output buffer and will already contain the dense gradient
		The output buffer x has d components and xq has d/2 components because one uint8_t stores two 4-bit values
	*/
	float2 *x2 = reinterpret_cast<float2*>(x); // we will read two values from x at once

	const LL B = gridDim.x; // number of blocks
	const LL Bid = blockIdx.x; // block id
// 	const LL T = blockDim.x; // number of threads
	const LL Tid = threadIdx.x; // thread id

	LL index = Bid * B + Tid; // this is the index used for x2 and xq
	LL d2 = d >> 1; // d2 will be the size of xq

	float u = (x_max - x_min) / 15.0f; // 15 = 16 - 1 = 2^4 - 1

	float msb; // the MSB of a xq component
	float lsb; // the LSB of a xq component
	float2 vx2; // the value that will store x2[index]
	uint8_t vq; // the value that will store xq[index]

	if(index < d2) {
		vx2 = x2[index];
		vq = xq[index];

		msb = (float)((vq & 0xF0) >> 4);
		lsb = (float)(vq & 0x0F);

        // += operation happens here
		vx2.x += msb * u + x_min;
		vx2.y += lsb * u + x_min;

		x2[index] = vx2;
	}

	if((d & 1) && (Tid == 0) && (Bid == 0)) { // do some more work on the first thread in the first block
	    // see the schematic in PhD #9 notebook, page 56
	    float vx = x[d - 1]; // read the last element from x which was left over after reading in chunks of 2 above
	    vq = xq[d2]; // read the last element in xq
	    msb = (uint8_t)((vq & 0xF0) >> 4); // we only need the MSB here
	    vx += msb * u + x_min;
	    x[d-1] = vx;
	}
}
void asymm_global_quant_inv_cuda(int blocks, int threads, LL d, torch::Tensor xq, float x_min, float x_max, torch::Tensor x) {
	asymm_global_quant_inv_kernel<<<blocks, threads>>>(
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
