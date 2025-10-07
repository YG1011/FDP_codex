#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 3D denoiser (for FDP): epsilon prediction on voxel grids.
point cloud -> occupancy voxel (optional smooth) -> add noise at t -> predict eps

This version applies performance improvements:
- Enable cuDNN benchmark & TF32 fast paths
- Optional AMP with fp16/bf16 (bf16 default, no GradScaler needed)
- channels_last_3d memory format
- DataLoader persistent workers + prefetch
- Remove per-iteration empty_cache()
- Optional gradient accumulation (--accum-steps)
- Lightweight profiling (--profile-interval)
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from util.misc import (
    THOUSAND, CheckpointManager, BlackHole,
    seed_all, get_new_log_dir, get_logger
)
from dataset.modelnet40 import ModelNet40
from purifier.voxelizer import pointcloud_to_occupancy, gaussian_smooth_occupancy
from models.denoiser3d import UNet3D, DDPM3D


# -------------------------------
# Arguments
# -------------------------------
parser = argparse.ArgumentParser("Train 3D epsilon-prediction denoiser for FDP (optimized)")

# --- FDP denoiser training args ---
parser.add_argument('--res', type=int, default=96, help='voxel resolution')
parser.add_argument('--T', type=int, default=1000, help='diffusion steps (training schedule)')
parser.add_argument('--beta-1', type=float, default=1e-4)
parser.add_argument('--beta-T', type=float, default=2e-2)
parser.add_argument('--base-ch', type=int, default=24, help='UNet3D base channels')

# Datasets and loaders
parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'])
parser.add_argument('--num-points', type=int, default=1024)
parser.add_argument('--train-batch_size', type=int, default=1)
parser.add_argument('--val-batch_size', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=8)

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight-decay', type=float, default=0.0)

# Training / logging
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log-root', type=str, default='./logs')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('--val-freq', type=int, default=1)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--save-path', type=str, default='models/pretrained/denoise3d.pt')

# 稳定/显存/性能
parser.add_argument('--amp', action='store_true', help='enable autocast (mixed precision)')
parser.add_argument('--amp-dtype', type=str, default='bf16', choices=['fp16', 'bf16'], help='dtype for autocast when --amp is enabled')
parser.add_argument('--max-grad-norm', type=float, default=1.0, help='gradient clipping')
parser.add_argument('--pre-smooth-sigma', type=float, default=0.8, help='gaussian smooth on occupancy (in voxels)')
parser.add_argument('--splat-ks', type=int, default=1, help='point->voxel splat kernel size')
parser.add_argument('--accum-steps', type=int, default=4, help='gradient accumulation steps (>=1)')
parser.add_argument('--profile-interval', type=int, default=100, help='log simple timing every N iters (0=off)')


def main(cli_args=None):

    args = parser.parse_args()
    seed_all(args.seed)

    # -------------------------------
    # Logging
    # -------------------------------
    if args.logging:
        log_dir = get_new_log_dir(
            args.log_root,
            prefix=f'fdp_denoiser/res{args.res}/',
            postfix=args.tag if args.tag is not None else ''
        )
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        ckpt_mgr = CheckpointManager(log_dir)
    else:
        logger = get_logger('train', None)
        writer = BlackHole()
        ckpt_mgr = BlackHole()
    logger.info(args)

    # -------------------------------
    # Data
    # -------------------------------
    if args.dataset != "modelnet40":
        raise Exception("Unavailable Dataset!")

    # 安全地加载数据集
    try:
        train_dset = ModelNet40(partition='train', scale_mode='none', num_points=args.num_points)
        val_dset   = ModelNet40(partition='test',  scale_mode='none', num_points=args.num_points)
        logger.info(f"成功加载数据集: 训练集 {len(train_dset)} 样本, 验证集 {len(val_dset)} 样本")
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        raise e

    pin_mem = (args.device == 'cuda')
    common_loader_kw = dict(
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=pin_mem,
    )
    # 更高数据吞吐
    if args.num_workers > 0:
        common_loader_kw.update(dict(persistent_workers=True, prefetch_factor=4))

    train_loader = DataLoader(
        train_dset, batch_size=args.train_batch_size, shuffle=True, **common_loader_kw
    )
    val_loader = DataLoader(
        val_dset, batch_size=args.val_batch_size, shuffle=False, **common_loader_kw
    )



    device = torch.device("cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu")

    # -------------------------------
    # CUDA/cuDNN: 高性能配置
    # -------------------------------
    if torch.cuda.is_available():
        # 一次性清理即可；不要在循环里频繁 empty_cache
        torch.cuda.empty_cache()

        # 高性能卷积/矩阵乘设置
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

        # 保留默认 allocator 行为；通常无需手动限流或 expandable_segments
        logger.info("CUDA/cuDNN 已配置为高性能模式 (TF32+benchmark)")

    # -------------------------------
    # Model
    # -------------------------------
    net3d  = UNet3D(in_ch=1, base_ch=args.base_ch, time_dim=256).to(device)
    net3d  = net3d.to(memory_format=torch.channels_last_3d)

    ddpm3d = DDPM3D(net3d, T=args.T, beta_1=args.beta_1, beta_T=args.beta_T).to(device)

    optimizer = optim.AdamW(net3d.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP
    use_amp = bool(args.amp)
    autocast_dtype = torch.bfloat16 if (use_amp and args.amp_dtype == 'bf16') else torch.float16
    # GradScaler 仅在 fp16 时需要；bf16 不用 scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and args.amp_dtype == 'fp16'))

    def autocast_ctx():
        return torch.amp.autocast('cuda', dtype=autocast_dtype, enabled=use_amp)

    # -------------------------------
    # Helpers
    # -------------------------------
    def _make_voxels(pc_bnc: torch.Tensor) -> torch.Tensor:
        """点云 -> 占据网格 (B,1,R,R,R)
        提示：如果该函数是 CPU 实现，建议未来搬到 Dataset 线程或改 GPU 实现。
        """
        V = pointcloud_to_occupancy(
            pc_bnc, res=args.res, bounds=((-1,1),(-1,1),(-1,1)), splat_ks=args.splat_ks
        )
        if not torch.isfinite(V).all():
            raise RuntimeError("Non-finite occupancy detected")

        sigma = float(args.pre_smooth_sigma)
        if sigma > 0:
            V = gaussian_smooth_occupancy(V, sigma_vox=sigma)
        return V


    def train_one_epoch(epoch: int):
        net3d.train(); ddpm3d.train()
        pbar = tqdm(train_loader, desc=f'[Train] epoch {epoch}/{args.epochs}')
        total_loss = 0.0; steps = 0

        accum_steps = max(1, int(args.accum_steps))
        if (use_amp and args.amp_dtype == 'fp16'):
            step_backward = lambda l: scaler.scale(l).backward()
            step_opt = lambda: (scaler.unscale_(optimizer),
                                (torch.nn.utils.clip_grad_norm_(net3d.parameters(), args.max_grad_norm) if args.max_grad_norm > 0 else None),
                                scaler.step(optimizer), scaler.update())
        else:
            step_backward = lambda l: l.backward()
            step_opt = lambda: ((torch.nn.utils.clip_grad_norm_(net3d.parameters(), args.max_grad_norm) if args.max_grad_norm > 0 else None),
                                optimizer.step())

        prof_t0 = time.time()

        for i, batch in enumerate(pbar):
            pc = batch['pointcloud'].to(device, non_blocking=True).float()  # (B,N,3)
            V  = _make_voxels(pc)                                           # (B,1,R,R,R)

            
            if not torch.isfinite(V).all():
                logger.warning("Non-finite voxel detected, skip this batch.")
                continue

            V  = V.to(device=device, dtype=torch.float32, non_blocking=True)
            V  = V.contiguous(memory_format=torch.channels_last_3d)

            B = V.size(0)
            t = torch.randint(1, args.T, (B,), device=device, dtype=torch.long)

            a_bar = ddpm3d.alphas_bar[t].view(B,1,1,1,1).to(dtype=V.dtype, device=V.device)
            eps   = torch.randn_like(V)
            x_t   = a_bar.sqrt()*V + (1.0 - a_bar).sqrt()*eps

            with autocast_ctx():
                loss = F.mse_loss(ddpm3d.predict_eps(x_t, t), eps)
                loss = loss / accum_steps

            optimizer.zero_grad(set_to_none=True) if (i % accum_steps == 0) else None
            step_backward(loss)

            if (i + 1) % accum_steps == 0:
                if (use_amp and args.amp_dtype == 'fp16'):
                    step_opt()
                else:
                    step_opt()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accum_steps
            steps += 1

            interval = getattr(args, 'profile_interval', 0)
            if interval and (i % max(1, interval) == 0) and i > 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.time() - prof_t0
                it_s = max(1e-9, i) / dt
                pbar.set_postfix(loss=f'{(total_loss/steps):.4f}', it_s=f'{it_s:.2f}')

        avg = total_loss / max(1, steps)
        writer.add_scalar('train/loss', avg, epoch)
        logger.info(f'[Train] epoch {epoch} | loss {avg:.6f}')
        return avg


    @torch.no_grad()
    def validate(epoch: int):
        net3d.eval(); ddpm3d.eval()
        total_loss = 0.0; steps = 0

        for i, batch in enumerate(tqdm(val_loader, desc=f'[Val] epoch {epoch}')):
            pc = batch['pointcloud'].to(device, non_blocking=True).float()
            V  = _make_voxels(pc)
            V  = V.to(device=device, dtype=torch.float32, non_blocking=True)
            V  = V.contiguous(memory_format=torch.channels_last_3d)

            B = V.size(0)
            t = torch.randint(1, args.T, (B,), device=device, dtype=torch.long)

            a_bar = ddpm3d.alphas_bar[t].view(B,1,1,1,1).to(dtype=V.dtype, device=V.device)
            eps   = torch.randn_like(V)
            x_t   = a_bar.sqrt()*V + (1.0 - a_bar).sqrt()*eps

            with autocast_ctx():
                loss = F.mse_loss(ddpm3d.predict_eps(x_t, t), eps)

            total_loss += loss.item()
            steps += 1

        avg = total_loss / max(1, steps)
        writer.add_scalar('val/loss', avg, epoch)
        logger.info(f'[Val] epoch {epoch} | loss {avg:.6f}')
        return avg


    # -------------------------------
    # Main loop
    # -------------------------------
    logger.info('Start training FDP 3D denoiser (optimized)...')
    best_val = float('inf')

    Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        _ = train_one_epoch(ep)

        if (ep % args.val_freq == 0) or (ep == args.epochs):
            val_loss = validate(ep)

            # 保存 ddpm 的权重，便于 eval_purify.py 使用： denoiser.load_state_dict(sd, strict=False)
            ckpt = {
                'state_dict': ddpm3d.state_dict(),    # 关键：包含 net 与调度 buffer
                'net_state_dict': net3d.state_dict(), # 额外备份纯网络
                'args': vars(args),
                'epoch': ep,
                'val_loss': val_loss,
            }
            torch.save(ckpt, args.save_path)
            logger.info(f'[Save] saved to {args.save_path}')

            # 维护一份“best”权重（兼容你项目里的 CheckpointManager 签名差异）
            if val_loss < best_val:
                best_val = val_loss
                try:
                    ckpt_mgr.save(net3d, args, best_val, step=ep)
                except TypeError:
                    try:
                        ckpt_mgr.save(net3d, args, best_val, ep)
                    except TypeError:
                        best_path = os.path.join(os.path.dirname(args.save_path), f'denoise3d_best_ep{ep:04d}.pt')
                        torch.save({
                            'epoch': ep,
                            'val_loss': best_val,
                            'state_dict': ddpm3d.state_dict(),
                            'net_state_dict': net3d.state_dict(),
                            'args': vars(args),
                        }, best_path)
                        logger.info(f'[Save:FALLBACK] saved best to {best_path}')

    logger.info('Training finished.')


if __name__ == "__main__":
    main()