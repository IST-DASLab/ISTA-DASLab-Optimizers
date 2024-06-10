clear
CUDA_VISIBLE_DEVICES=7 python example.py \
    --model rn18 \
    --dataset_path ./data \
    --dataset_name cifar10 \
    --optimizer sparse-mfac \
    \
    --epochs 100 \
    --batch_size 128 \
    \
    --lr 1e-3 \
    --damp 1e-6 \
    --m 1024 \
    --k 0.01 \
    \
    --precision bf16 \
    \
    --wandb_project ista-daslab-optimizers \
    --wandb_group example \
    --wandb_job_type cifar10 \
    --wandb_name sparse-mfac
