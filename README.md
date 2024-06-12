# ISTA DAS Lab Optimization Algorithms Package
This repository contains optimization algorithms for Deep Learning developed by 
the Distributed Algorithms and Systems lab at Institute of Science and Technology Austria.

## Project status
- **June 5th, 2024**:
  - *DONE*: the project is locally installable via `pip install .` 
  - *NEXT*:
    - working on examples for Sparse M-FAC and Dense M-FAC
- **May 27th, 2024**:
  - we are currently working on solving the issues with the installation via `pip`. 

### Installation
We provide a script `install.sh` that creates a new environment, installs requirements 
and then builds the optimizers project. First of all, you have to clone this repository, then 
run the installation script.
```shell
git clone git@github.com:IST-DASLab/ISTA-DASLab-Optimizers.git
cd ISTA-DASLab-Optimizers
source install.sh
```

[//]: # (### ⚠️ Important Notice ⚠️)

[//]: # (We noticed it is useful to compile the kernels for each individual CUDA capability separately. For example, for CUDA capability &#40;CC&#41; 8.6, )

[//]: # (the CUDA kernels for `MicroAdam` will be installed in the package `micro_adam_sm86`, while for CC 9.0 it will be installed in the package)

[//]: # (`micro_adam_sm90`. Please install this library for each system where the CC is different to cover all possible cases for your system. The )

[//]: # (code will automatically detect the CC version and import the correct package if installed, otherwise will throw an error. The code that )

[//]: # (dynamically detects the CC version can be found )

[//]: # ([here]&#40;https://github.com/IST-DASLab/ISTA-DASLab-Optimizers/blob/main/ista_daslab_optimizers/tools.py#L17&#41;.)

## How to use optimizers?

We provide a minimal working example with ResNet-18 and CIFAR-10 for optimizers `micro_adam`, `acdc`, `sparse_mfac`, `dense_mfac`:
```shell
OPTIMIZER=micro_adam # or any other optimizer listed above
bash run_${OPTIMIZER}.sh
```

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
)

# from now on, you can use the variable `optimizer` as any other PyTorch optimizer
```