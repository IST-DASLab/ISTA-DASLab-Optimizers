clear

CUDA_VISIBLE_DEVICES=0 python example.py \
    --model rn18 \
    --dataset_path ./data \
    --dataset_name cifar10 \
    --optimizer micro-adam \
    \
    --epochs 100 \
    --batch_size 128 \
    \
    --lr 1e-3 \
    --m 10 \
    --k 0.01 \
    --ef_quant_bucket_size 32 \
    \
    --precision bf16 \
    \
    --wandb_project ista-daslab-optimizers \
    --wandb_group example \
    --wandb_job_type cifar10 \
    --wandb_name micro-adam