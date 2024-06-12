clear
CUDA_VISIBLE_DEVICES=0 python example.py \
    --model rn20 \
    --dataset_path ./data \
    --dataset_name cifar10 \
    --optimizer dense-mfac \
    \
    --epochs 100 \
    --batch_size 128 \
    \
    --lr 1e-3 \
    --damp 1e-6 \
    --m 1024 \
    \
    --wandb_project ista-daslab-optimizers \
    --wandb_group example \
    --wandb_job_type cifar10 \
    --wandb_name dense-mfac
