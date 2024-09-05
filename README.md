# ISTA DAS Lab Optimization Algorithms Package
This repository contains optimization algorithms for Deep Learning developed by 
the Distributed Algorithms and Systems lab at Institute of Science and Technology Austria.

The repository contains code for the following optimizers published by DASLab @ ISTA:
- **AC/DC**:
  - paper: [AC/DC: Alternating Compressed/DeCompressed Training of Deep Neural Networks](https://arxiv.org/abs/2106.12379)
  - official repository: [GitHub](https://github.com/IST-DASLab/ACDC)
- **M-FAC**:
  - paper: [M-FAC: Efficient Matrix-Free Approximations of Second-Order Information](https://arxiv.org/abs/2107.03356)
  - official repository: [GitHub](https://github.com/IST-DASLab/M-FAC)
- **Sparse M-FAC with Error Feedback**:
  - paper: [Error Feedback Can Accurately Compress Preconditioners](https://arxiv.org/abs/2306.06098)
  - official repository: [GitHub](https://github.com/IST-DASLab/EFCP/)
- **MicroAdam**:
  - paper: [MicroAdam: Accurate Adaptive Optimization with Low Space Overhead and Provable Convergence](https://arxiv.org/abs/2405.15593)
  - official repository: [GitHub](https://github.com/IST-DASLab/MicroAdam)

### Installation
To use the latest stable version of the repository, you can install via pip:

```shell
pip3 install ista-daslab-optimizers
```

We also provide a script `install.sh` that creates a new environment, installs requirements
and then installs the project as a Python package following these steps:

```shell
git clone git@github.com:IST-DASLab/ISTA-DASLab-Optimizers.git
cd ISTA-DASLab-Optimizers
source install.sh
```

## How to use optimizers?

In this repository we provide a minimal working example for CIFAR-10 for optimizers `acdc`, `dense_mfac`, `sparse_mfac` and `micro_adam`:
```shell
cd examples/cifar10
OPTIMIZER=micro_adam # or any other optimizer listed above
bash run_${OPTIMIZER}.sh
```

To integrate the optimizers into your own pipeline, you can use the following snippets:

### MicroAdam optimizer
```python
from ista_daslab_optimizers import MicroAdam

model = MyCustomModel()

optimizer = MicroAdam(
    model.parameters(), # or some custom parameter groups
    m=10, # sliding window size (number of gradients)
    lr=1e-5, # change accordingly
    quant_block_size=100_000, # 32 or 64 also works
    k_init=0.01, # float between 0 and 1 meaning percentage: 0.01 means 1%
    alpha=0, # 0 means sparse update and 0 < alpha < 1 means we integrate fraction alpha from EF to update and then delete it
)

# from now on, you can use the variable `optimizer` as any other PyTorch optimizer
```

# Versions summary:

---
- **1.1.3** @ September 5th, 2024:
  - allow using `SparseCoreMFACwithEF` separately by importing it in `sparse_mfac.__init__.py`
- **1.1.2** @ August 1st, 2024:
  - ***[1.1.0]:*** added support to densify the final update: introduced parameter alpha that controls the fraction of error feedback
  (EF) to be integrated into the update to make it dense. Finally, the fraction alpha will be discarded from the EF at
  the expense of another call to `Qinv` and `Q` (and implicitly quantization statistics computation).
  - ***[1.0.2]:*** added FSDP-compatible implementation by initializing the parameter states in the `update_step` method
  instead of MicroAdam constructor

- **1.0.1** @ June 27th, 2024:
  - removed version in dependencies to avoid conflicts with llm-foundry

- **1.0.0** @ June 20th, 2024:
  - changed minimum required Python version to 3.8+ and torch to 2.3.0+

- **0.0.1** @ June 13th, 2024:
  - added initial version of the package for Python 3.9+ and torch 2.3.1+