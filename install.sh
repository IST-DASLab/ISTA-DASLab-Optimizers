ENV_NAME=ista

#echo ">>>>> Loading CUDA 12.2 module"
#module load cuda/12.2

echo ">>>>> Creating environment \"${ENV_NAME}\""
conda create --name $ENV_NAME python=3.9 -y
conda activate $ENV_NAME

#echo ">>>>> Installing required packages..."
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#pip3 install wandb gpustat timm einops psutil

echo ">>>>> Installing ISTA-DASLab-Optimizers..."
rm -rf build dist *egg* # delete files from previous installations
pip3 install .