#!/bin/bash

module load cuda/12.2
conda create --name ist_daslab_optimizers python=3.9 -y
conda activate ist_daslab_optimizers
pip3 install torch torchvision torchaudio
pip3 install wandb gpustat
cd ~
git clone git@github.com:IST-DASLab/ISTA-DASLab-Optimizers.git
cd ISTA-DASLab-Optimizers
pip3 install .