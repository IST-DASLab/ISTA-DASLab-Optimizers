import warnings
warnings.filterwarnings("ignore")
import psutil
import os
import argparse
import math
import time
import numpy as np
import random
import torch
import gpustat
import wandb

from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from torchvision.models import resnet18
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from timm.optim.optim_factory import param_groups_weight_decay
from tqdm import tqdm
from ista_daslab_optimizers import MicroAdam, SparseMFAC, DenseMFAC

def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', type=str, default=None, help='The wandb entity')
    parser.add_argument('--wandb_project', type=str, required=True, help='The wandb project inside "ist" owner.')
    parser.add_argument('--wandb_group', type=str, required=True, help='The wandb group in the project.')
    parser.add_argument('--wandb_job_type', type=str, default=None, required=True, help='The wandb job type')
    parser.add_argument('--wandb_name', type=str, required=True, default=None, help='The name for the experiment in wandb runs')
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        choices=['rn18'],
                        help='Model to train')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset to use for training.')
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        choices=['cifar10'],
                        help='Dataset name')
    parser.add_argument('--optimizer',
                        type=str,
                        required=True,
                        choices=['sgd', 'adamw', 'micro-adam', 'sparse-mfac', 'dense-mfac', 'acdc'],
                        help='Optimizer to use for training')
    parser.add_argument('--epochs', type=int, required=True, help='The number of epochs to train the model for')
    parser.add_argument('--batch_size', type=int, required=True, help='Batchsize to use for training.')
    parser.add_argument('--lr', type=float, default=1e-3, required=True, help='Learning rate to use for training.')
    parser.add_argument('--damp', type=float, default=1e-6, required=False, help='Dampening for Sparse M-FAC')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum to use for training.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay to use for training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--m', type=int, required=True, help='Window size for micro-adam, sparse-mfac')
    parser.add_argument('--k', type=float, required=True, help='Gradient density for micro-adam, sparse-mfac')
    parser.add_argument('--ef_quant_bucket_size', type=int, default=0, help='Bucket size used for EF quantization')
    parser.add_argument('--precision', type=str, default='bf16', help='Data type to convert the model to')

    return parser.parse_args()

def get_ram_mem_usage():
    return round(psutil.Process().memory_info().rss / (2 ** 30), 2)

def get_gpu_mem_usage():
    gpus = gpustat.new_query().gpus
    gids = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    gpu_mem = sum([int(proc['gpu_memory_usage']) for gid in gids for proc in gpus[gid]['processes'] if int(proc['pid']) == os.getpid()])
    return gpu_mem

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def get_model(model_name, dataset_name, precision):
    num_classes = dict(cifar10=10)[dataset_name]
    model = None
    if model_name == 'rn18':
        model = resnet18(pretrained=False, num_classes=num_classes)

    if model is None:
        raise RuntimeError(f'Note sure how to build model {model_name}')

    if precision in ['bf16', 'bfloat16']:
        model = model.to(dtype=torch.bfloat16)

    return model

def get_optimizer(args, model):
    param_groups = param_groups_weight_decay(model=model, weight_decay=args.weight_decay)
    if args.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == 'micro-adam':
        return MicroAdam(
            param_groups,  # or some custom parameter groups
            m=args.m,  # sliding window size (number of gradients)
            lr=args.lr,  # change accordingly
            quant_block_size=args.ef_quant_bucket_size,  # 32 or 64 also works
            k_init=args.k)  # float between 0 and 1 meaning percentage: 0.01 means 1%
    if args.optimizer == 'sparse-mfac':
        return SparseMFAC(
            param_groups,
            lr=args.lr,
	        weight_decay=args.weight_decay,
            m=args.m,
            damp=args.damp,
            k_init=args.k,
            use_bf16=(args.precision == 'bf16'))
    if args.optimizer == 'dense-mfac':
        return DenseMFAC(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            ngrads=args.m,
            damp=args.damp)
    raise RuntimeError(f'Not sure how to build optimizer {args.optimizer_name}')

def get_cifar10_datasets(data_dir):
    _CIFAR10_RGB_MEANS = (0.491, 0.482, 0.447)
    _CIFAR10_RGB_STDS = (0.247, 0.243, 0.262)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR10_RGB_MEANS, std=_CIFAR10_RGB_STDS)
    ])
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR10_RGB_MEANS, std=_CIFAR10_RGB_STDS)
    ])
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset

def get_datasets(dataset_name, dataset_path):
    if dataset_name == 'cifar10':
        return get_cifar10_datasets(dataset_path)
    raise RuntimeError(f'Dataset {dataset_name} is currently not supported!')

def schedule_linear(base_lr, step, total_steps):
    return base_lr * (1 - step / total_steps)

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


@torch.no_grad()
def test(model, data, precision):
    loss, correct, test_dataset_size = 0, 0, 0
    progress = tqdm(data)
    progress.set_description('Evaluating...')
    for x, y in progress:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        if precision in ['bf16', 'bfloat16']:
            x = x.to(dtype=torch.bfloat16)

        y_hat = model(x)
        test_dataset_size += x.shape[0]

        loss += cross_entropy(y_hat, y, reduction='sum').item()
        pred = y_hat.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item()
    return correct, loss, test_dataset_size

def beautify_time(ss):
    ss = math.ceil(ss)
    ss = int(ss)
    hh = ss // 3600
    ss %= 3600
    mm = ss // 60
    ss %= 60
    return f'{hh}h {mm}m {ss}s'


def setup_wandb(args):
    return wandb.init(
        project=args.wandb_project,
        job_type=args.wandb_job_type,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_name,
        config=args,
        settings=wandb.Settings(start_method='fork'))

def main():
    args = get_arg_parse()
    setup_wandb(args)
    set_all_seeds(args.seed)
    model = get_model(args.model, args.dataset_name, args.precision).to('cuda:0')
    optimizer = get_optimizer(args, model)
    train_data, test_data = get_datasets(args.dataset_name, args.dataset_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    steps = 0
    total_steps = args.epochs * math.ceil(len(train_data) / args.batch_size)

    for epoch in range(1, args.epochs + 1):
        train_start = time.time()

        train_loss, train_size, train_correct = 0., 0, 0
        progress = tqdm(train_loader)

        for x, y in progress:
            steps += 1
            lr = schedule_linear(args.lr, steps, total_steps)
            set_lr(optimizer, lr)

            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            if args.precision in ['bf16', 'bfloat16']:
                x = x.to(dtype=torch.bfloat16)

            crt_batch_size = x.shape[0]
            train_size += crt_batch_size

            optimizer.zero_grad(set_to_none=True)

            y_hat = model(x)
            loss = cross_entropy(y_hat, y).cuda()
            loss.backward()
            optimizer.step()

            step_loss = loss.item() * x.size(0)  # the loss for the current batch
            train_loss += step_loss  # the loss for the epoch

            train_correct += (torch.argmax(y_hat, 1) == y).sum().item()

            loss_str = f'[Epoch {epoch}/{args.epochs}] Loss Batch={step_loss / crt_batch_size:.2f} | Loss Epoch={train_loss / train_size:.2f}'
            progress.set_description(f'{loss_str}')
        # end epoch
        train_elapsed = time.time() - train_start

        test_start = time.time()
        model.eval()
        test_correct, test_loss, test_size = test(model, test_loader, args.precision)
        model.train()
        test_elapsed = time.time() - test_start

        train_loss /= train_size
        train_accuracy = round(train_correct / train_size * 100, 2)
        test_loss /= test_size
        test_accuracy = round(test_correct / test_size * 100, 2)

        print(f'Loss Train/Test:    \t{train_loss:.4f} / {test_loss:.4f}\n'
              f'Accuracy Train/Test:\t{train_accuracy:.2f}% / {test_accuracy:.2f}%\n'
              f'Elapsed Train/Test: \t{beautify_time(train_elapsed)} / {beautify_time(test_elapsed)}\n'
              f'Current/Base LR:    \t{lr} / {args.lr}\n')

        wandb.log({
            'epoch/epoch': epoch,
            'epoch/train_loss': train_loss,
            'epoch/train_accuracy': train_accuracy,
            'epoch/test_loss': test_loss,
            'epoch/test_accuracy': test_accuracy,
            'epoch/train_elapsed': train_elapsed,
            'epoch/test_elapsed': test_elapsed,
            'epoch/gpu_mem_usage': get_gpu_mem_usage(),
            'epoch/ram_mem_usage': get_ram_mem_usage(),
            'epoch/lr': lr,
        })


if __name__ == '__main__':
    main()
