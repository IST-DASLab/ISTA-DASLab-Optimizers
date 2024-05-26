# ISTA DAS Lab Optimization Algorithms Package
This repository contains optimization algorithms for Deep Learning developed by 
the Distributed Algorithms and Systems lab at Institute of Science and Technology Austria.

### Installation
We provide a script `install.sh` that creates a new environment, installs requirements 
and then builds the optimizers project. We use `python3 setup.py install` because we are 
currently facing issues with installing the extension as a PyPi package, but this will be
fixed as soon as possible.
```shell
bash install.sh
```

## How to use optimizers?

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