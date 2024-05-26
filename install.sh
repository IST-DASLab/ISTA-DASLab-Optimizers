ENV_NAME=ista_daslab_optimizers

echo ">>>>> Loading CUDA 12.2 module"
module load cuda/12.2

echo ">>>>> Creating environment \"${ENV_NAME}\""
conda create --name $ENV_NAME python=3.9 -y

echo ">>>>> Activating environment"
conda activate $ENV_NAME

echo ">>>>> Installing packages..."
pip3 install torch torchvision torchaudio wandb gpustat

echo ">>>>> Installing packages..."
pip3 install .