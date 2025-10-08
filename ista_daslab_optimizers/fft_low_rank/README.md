# [FFT-based Dynamic Subspace Selection for Low-Rank Adaptive Optimization of Large Language Models](https://arxiv.org/pdf/2505.17967)

This repository contains the source code for Trion and DCT-AdamW.

### Intro
We propose a cheaper alternative to low-rank projections based on
SVD and Subspace Iteration. We propose coupling our **Dynamic 
Column Selection** approach with the orthogonal matrix of the 
Discrete Cosine Transform (DCT) matrix to efficiently compute a 
low-rank projection of gradient/momentum in the context
of adaptive gradient optimizers.

### Dynamic Column Selection Approach
We propose this technique to dynamically select $r$ columns from a
predefined orthogonal matrix to minimize the projection error. To 
compute a low-rank projection $g$ of a matrix $G$ using the orthogonal
matrix $Q$, we first compute a similarity matrix $S = GQ$ and then
choose the largest $r$ columns in $S$ based on their $\ell_1$-norm,
which we call $i_t$. We then obtain the actual low-rank projection
matrix $P = Q[:, i_t]$ (simply indexing the matrix $Q$) and we the
low-rank gradient $g = S[:, i_t]$.

This approach works with any orthogonal matrix. However, certain matrices
in particular exhibit some properties that help us reduce the overhead of
the matrix multiplication $S = GQ$.

### Why DCT?
While any orthogonal matrix works with the **Dynamic Column Selection**
approach, we choose the orthogonal matrix from the Discrete Cosine Transform
(DCT). When $Q$ is the DCT matrix, the matrix $S = GQ$ is called the DCT of
$G$. Since $Q_{ij} = \sqrt{2/n} \cdot \cos \frac{i(2j+1)\pi}{2n}$, we can
reduce the complexity of computing $S=GQ$ from $O(n^3)$ to $O(n^2\log(n))$
using Makhoul's $N$-point algorithm that uses **Fast Fourier Transform (FFT)**
under the hood.

### Trion
We integrate the Dynamic Column Selection approach with DCT projection in the
existing Dion optimizer to replace the inacurate Subspace Iteration that
uses the expensive QR-decomposition approach whose running time is dependent
on the chosen rank. We compute a low-rank momentum using DCT and replace the
QR-decomposition with the Newton-Schulz iteration introduced in Muon.

### DCT-AdamW
We integrate the Dynamic Column Selection approach with DCT projection to
replace the SVD projection in FRUGAL and FIRA optimizers to compute a low-rank
projection of the gradient, which is then sent to the AdamW optimizer. The
resulting optimizers are called DCT-FRUGAL and DCT-FIRA.

We also propose a standalone AdamW with low-rank optimizer states which we call
DCT-AdamW.