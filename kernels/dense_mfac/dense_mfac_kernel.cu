#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int SIZE = 32;
const int MAX = 1024;

template <typename scalar_t>
__device__ __forceinline__ scalar_t GetElement(
  const scalar_t* __restrict__ A, int m,
  int row, int col
) {
  return A[row * m + col];
}

template <typename scalar_t>
__device__ __forceinline__ void SetElement(
  scalar_t* __restrict__ A, int m, 
  int row, int col, 
  scalar_t value
) {
  A[row * m + col] = value;
}

/* Kernel for computing `coef` (required setup & update operations) */

template <typename scalar_t>
__global__ void HinvCoefKernelDiag(
    int m,
    const scalar_t* __restrict__ tmp,
    scalar_t* __restrict__ coef
);

template <typename scalar_t>
__global__ void HinvCoefKernelMain(
  int m,
  const scalar_t* __restrict__ tmp,
  scalar_t* __restrict__ coef,
  int iter 
);

// NOTE: for simplicity, we assume that `m` is always divisible by `SIZE` 
torch::Tensor hinv_setup_cuda(torch::Tensor tmp, torch::Tensor coef) {
  const auto m = tmp.size(0);
  const dim3 threads(SIZE, SIZE);
  const dim3 blocks(m / SIZE, m / SIZE);

  AT_DISPATCH_FLOATING_TYPES(tmp.type(), "hinv_setup_cuda", ([&] {
      HinvCoefKernelDiag<scalar_t><<<m / SIZE, threads>>>(
        m, tmp.data<scalar_t>(), coef.data<scalar_t>()
      );
    })
  );
  for (int i = 0; i < m / SIZE - 1; i++) {
    AT_DISPATCH_FLOATING_TYPES(tmp.type(), "hinv_setup_cuda", ([&] {
        HinvCoefKernelMain<scalar_t><<<blocks, threads>>>(
          m, tmp.data<scalar_t>(), coef.data<scalar_t>(), i 
        );
      })
    );
  }

  return coef;
}

template <typename scalar_t>
__global__ void HinvCoefKernelMain(
    int m,
    const scalar_t* __restrict__ tmp,
    scalar_t* __restrict__ coef,
    int iter
) {
  // one thread per block element

  // top-left of target block
  int toi = blockIdx.x * SIZE;
  int toj = blockIdx.y * SIZE;
  // top-left of source block
  int fromi = (blockIdx.y + iter) * SIZE;
  int fromj = toj;

  // only compute below (current) diagonal
  if (fromi >= toi)
    return;

  // current relative position
  int x = threadIdx.x;
  int y = threadIdx.y;
  // current absolute position
  int i = toi + x;
  int j = toj + y;

  // parallel load relevant blocks of `coef` and `tmp` 
  __shared__ scalar_t from_coef[SIZE][SIZE];
  __shared__ scalar_t to_tmp[SIZE][SIZE];
  from_coef[x][y] = GetElement(coef, m, fromi + x, fromj + y);
  to_tmp[x][y] = GetElement(tmp, m, i, fromi + y); 
  __syncthreads();

  // parallel matmul 
  scalar_t res = GetElement(coef, m, i, j);
  for (int k = 0; k < SIZE; k++)
    res += to_tmp[x][k] * from_coef[k][y];
  SetElement(coef, m, i, j, res);

  // keep only next sequential blocks 
  if (toi != fromi + SIZE)
    return;
  __syncthreads();

  // parallel load relevant blocks of `coef` and `tmp` 
  from_coef[x][y] = GetElement(coef, m, i, j);
  to_tmp[x][y] = GetElement(tmp, m, i, toi + y);
  __syncthreads();

  // parallel sequential vector-matrix multiplies
  res = from_coef[x][y];
  for (int k = 0; k < SIZE; k++) {
    if (k < x)
      res += to_tmp[x][k] * from_coef[k][y];
    if (k == x - 1) {
      // parallel write block row
      from_coef[x][y] = res;
      SetElement(coef, m, i, j, res);
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void HinvCoefKernelDiag(
    int m,
    const scalar_t* __restrict__ tmp,
    scalar_t* __restrict__ coef
) {
  // one thread per block element

  // current relative position
  int x = threadIdx.x;
  int y = threadIdx.y;
  // current absolute position
  int i = blockIdx.x * SIZE + x;
  int j = blockIdx.x * SIZE + y;

  // parallel load relevant blocks of `coef` and `tmp`
  __shared__ scalar_t from_coef[SIZE][SIZE];
  __shared__ scalar_t to_tmp[SIZE][SIZE];
  from_coef[x][y] = GetElement(coef, m, i, j);
  to_tmp[x][y] = GetElement(tmp, m, i, j);
  __syncthreads();

  // parallel sequential vector-matrix multiplies
  scalar_t res = 0;
  for (int k = 0; k < SIZE; k++) {
    if (k < x)
      res += to_tmp[x][k] * from_coef[k][y];
    // don't write diagonal elements
    if (k == x - 1 && x != y) {
      // parallel write block row
      from_coef[x][y] = res;
      SetElement(coef, m, i, j, res);
    }
    __syncthreads();
  }
}


/* Kernel for computing `giHix` (required for multiplication) */ 

template <typename scalar_t>
__global__ void HinvMulKernel(
	int rows,
    int m,
    const scalar_t* __restrict__ giHig,
    scalar_t* __restrict__ giHix
);

// NOTE: currently only works for `m` <= 1024
torch::Tensor hinv_mul_cuda(int rows, torch::Tensor giHig, torch::Tensor giHix) {
  const auto m = giHig.size(0);
  AT_DISPATCH_FLOATING_TYPES(giHig.type(), "hinv_mul_cuda", ([&] {
      HinvMulKernel<scalar_t><<<1, m>>>(
        rows, m, giHig.data<scalar_t>(), giHix.data<scalar_t>()
      );
    })
  );
  return giHix;
}

template <typename scalar_t>
__global__ void HinvMulKernel(
	int rows,
    int m,
    const scalar_t* __restrict__ giHig,
    scalar_t* __restrict__ giHix
) {
    int i = threadIdx.x;

    // parallel load relevant coefficients from `giHix` and `giHig`
    __shared__ scalar_t denom[MAX];
    __shared__ scalar_t tmp[MAX];
    denom[i] = GetElement(giHig, m, i, i) + rows; // changed by ionut: fix_scaling
    tmp[i] = GetElement(giHix, m, 0, i); 
    __syncthreads();

    // compute parallel sequential linear combination 
    for (int j = 1; j < m; j++) {
        if (j <= i) {
			scalar_t sub = GetElement(giHig, m, j - 1, i) * tmp[j - 1] / denom[j - 1];
			tmp[i] -= sub;
		}
		__syncthreads();
    }

    SetElement(giHix, m, 0, i, tmp[i]);
}
