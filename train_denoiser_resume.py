#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 3D denoiser (for FDP): epsilon prediction on voxel grids.
point cloud -> occupancy voxel (optional smooth) -> add noise at t -> predict eps

This version adds **clean resume support** and keeps the performance tweaks:
- cuDNN benchmark + TF32 fast paths
- Optional AMP (bf16/fp16)
- channels_last_3d memory format
- DataLoader persistent workers + prefetch
- Gradient accumulation (--accum-steps)
- Lightweight profiling (--profile-interval)
- **NEW**: --resume / --resume-optim / --resume-strict
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.tensorboard
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
parser = argparse.ArgumentParser("Train 3D epsilon-prediction denoiser for FDP (optimized + clean resume)")

# FDP / model
parser.add_argument('--res', type=int, default=96, help='voxel resolution')
parser.add_argument('--T', type=int, default=1000, help='diffusion steps (training schedule)')
parser.add_argument('--beta-1', type=float, default=1e-4)
parser.add_argument('--beta-T', type=float, default=2e-2)
parser.add_argument('--base-ch', type=int, default=24, help='UNet3D base channels')

# Dataset / loader
parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'])
parser.add_argument('--num-points', type=int, default=1024)
parser.add_argument('--train-batch_size', type=int, default=4)
parser.add_argument('--val-batch_size', type=int, default=4)
parser.add_argument('--num-workers', type=int, default=8)

# Optimizer
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight-decay', type=float, default=0.0)

# Train / log
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log-root', type=str, default='./logs')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('--val-freq', type=int, default=1)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--save-path', type=str, default='models/pretrained/denoise3d.pt')

# Stability / perf
parser.add_argument('--amp', action='store_true', help='enable autocast (mixed precision)')
parser.add_argument('--amp-dtype', type=str, default='bf16', choices=['fp16', 'bf16'], help='dtype for autocast when --amp is enabled')
parser.add_argument('--max-grad-norm', type=float, default=1.0, help='gradient clipping; 0 to disable')
parser.add_argument('--pre-smooth-sigma', type=float, default=0.8, help='gaussian smooth on occupancy (in voxels)')
parser.add_argument('--splat-ks', type=int, default=1, help='point->voxel splat kernel size')
parser.add_argument('--accum-steps', type=int, default=4, help='gradient accumulation steps (>=1)')
parser.add_argument('--profile-interval', type=int, default=100, help='log simple timing every N iters (0=off)')

# Resume
parser.add_argument('--resume', type=str, default='/home/linux/Documents/FDP/models/pretrained/45epoch_denoise3d.pt', help='path to checkpoint .pt to resume from (e.g., /home/linux/Documents/FDP/models/pretrained/45epoch_denoise3d.pt)')
parser.add_argument('--resume-optim', action='store_true', help='also resume optimizer/GradScaler states if available')
parser.add_argument('--resume-strict', action='store_true', help='strictly match keys when loading state_dicts')


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
    if args.dataset != 'modelnet40':
        raise ValueError('Unavailable Dataset')

    try:
        train_dset = ModelNet40(partition='train', scale_mode='none', num_points=args.num_points)
        val_dset   = ModelNet40(partition='test',  scale_mode='none', num_points=args.num_points)
        logger.info(f"成功加载数据集: 训练集 {len(train_dset)} 样本, 验证集 {len(val_dset)} 样本")
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        raise

    pin_mem = (args.device == 'cuda')
    common_loader_kw = dict(num_workers=args.num_workers, drop_last=False, pin_memory=pin_mem)
    if args.num_workers > 0:
        common_loader_kw.update(dict(persistent_workers=True, prefetch_factor=4))

    train_loader = DataLoader(train_dset, batch_size=args.train_batch_size, shuffle=True, **common_loader_kw)
    val_loader   = DataLoader(val_dset,   batch_size=args.val_batch_size,   shuffle=False, **common_loader_kw)

    # -------------------------------
    # Device & CUDA setup
    # -------------------------------
    device = torch.device('cuda' if (torch.cuda.is_available() and args.device == 'cuda') else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # one-time clear
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        logger.info('CUDA/cuDNN 已配置为高性能模式 (TF32+benchmark)')

    # -------------------------------
    # Model / DDPM wrapper
    # -------------------------------
    net3d  = UNet3D(in_ch=1, base_ch=args.base_ch, time_dim=256).to(device)
    net3d  = net3d.to(memory_format=torch.channels_last_3d)
    ddpm3d = DDPM3D(net3d, T=args.T, beta_1=args.beta_1, beta_T=args.beta_T).to(device)

    optimizer = optim.AdamW(net3d.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP setup
    use_amp = bool(args.amp)
    autocast_dtype = torch.bfloat16 if (use_amp and args.amp_dtype == 'bf16') else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and args.amp_dtype == 'fp16'))  # bf16 不需要 scaler

    def autocast_ctx():
        return torch.amp.autocast('cuda', dtype=autocast_dtype, enabled=use_amp)

    # -------------------------------
    # Resume
    # -------------------------------
    start_epoch = 45
    best_val = float('inf')
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f'--resume path not found: {args.resume}')
        logger.info(f'[Resume] loading checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)

        # Consistency hints
        cargs = ckpt.get('args', {})
        warn = []
        for k in ['T', 'beta_1', 'beta_T', 'res', 'base_ch']:
            old = cargs.get(k, None)
            new = getattr(args, k) if hasattr(args, k) else None
            if (old is not None) and (new is not None) and old != new:
                warn.append(f'{k}: ckpt={old} != args={new}')
        if warn:
            logger.warning('[Resume] 参数与检查点不一致，请确认: ' + '; '.join(warn))

        # Prefer full ddpm3d (contains buffers); fallback to raw net
        strict_flag = bool(args.resume_strict)
        loaded = False
        if 'state_dict' in ckpt:
            try:
                ddpm3d.load_state_dict(ckpt['state_dict'], strict=strict_flag)
                loaded = True
                logger.info(f'[Resume] ddpm3d state_dict loaded (strict={strict_flag})')
            except Exception as e:
                logger.warning(f'[Resume] ddpm3d.load_state_dict failed: {e}')
        if not loaded and ('net_state_dict' in ckpt):
            net3d.load_state_dict(ckpt['net_state_dict'], strict=False)
            logger.info('[Resume] Fallback: loaded net3d weights only (strict=False)')

        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_val = float(ckpt.get('val_loss', float('inf')))

        # Optionally resume optimizer & scaler
        if args.resume_optim:
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    logger.info('[Resume] optimizer state restored.')
                except Exception as e:
                    logger.warning(f'[Resume] optimizer.load_state_dict failed: {e}')
            if 'scaler_state_dict' in ckpt and (use_amp and args.amp_dtype == 'fp16'):
                try:
                    scaler.load_state_dict(ckpt['scaler_state_dict'])
                    logger.info('[Resume] GradScaler state restored.')
                except Exception as e:
                    logger.warning(f'[Resume] scaler.load_state_dict failed: {e}')

        logger.info(f'[Resume] start_epoch={start_epoch}, historical_best_val={best_val:.6f}')

    # -------------------------------
    # Helpers
    # -------------------------------
    def _make_voxels(pc_bnc: torch.Tensor) -> torch.Tensor:
        """点云 -> 占据网格 (B,1,R,R,R)"""
        V = pointcloud_to_occupancy(
            pc_bnc, res=args.res, bounds=((-1, 1), (-1, 1), (-1, 1)), splat_ks=args.splat_ks
        )
        if not torch.isfinite(V).all():
            raise RuntimeError('Non-finite occupancy detected')
        sigma = float(args.pre_smooth_sigma)
        if sigma > 0:
            V = gaussian_smooth_occupancy(V, sigma_vox=sigma)
        return V

    def train_one_epoch(epoch: int):
        net3d.train(); ddpm3d.train()
        pbar = tqdm(train_loader, desc=f'[Train] epoch {epoch}/{args.epochs}')
        total_loss = 0.0
        steps = 0
        accum_steps = max(1, int(args.accum-steps)) if hasattr(args, 'accum-steps') else max(1, int(args.accum_steps))

        def _clip():
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net3d.parameters(), args.max_grad_norm)

        for i, batch in enumerate(pbar):
            pc = batch['pointcloud'].to(device, non_blocking=True).float()  # (B,N,3)
            V  = _make_voxels(pc)                                           # (B,1,R,R,R)
            if not torch.isfinite(V).all():
                continue
            V  = V.to(device=device, dtype=torch.float32, non_blocking=True)
            V  = V.contiguous(memory_format=torch.channels_last_3d)

            B = V.size(0)
            t = torch.randint(1, args.T, (B,), device=device, dtype=torch.long)
            a_bar = ddpm3d.alphas_bar[t].view(B, 1, 1, 1, 1).to(dtype=V.dtype, device=V.device)
            eps   = torch.randn_like(V)
            x_t   = a_bar.sqrt() * V + (1.0 - a_bar).sqrt() * eps

            if (i % accum_steps) == 0:
                optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                loss = F.mse_loss(ddpm3d.predict_eps(x_t, t), eps)
                loss = loss / accum_steps

            if use_amp and args.amp_dtype == 'fp16':
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # step on boundary
            if ((i + 1) % accum_steps) == 0:
                if use_amp and args.amp_dtype == 'fp16':
                    scaler.unscale_(optimizer)
                    _clip()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    _clip()
                    optimizer.step()

            total_loss += loss.item() * accum_steps
            steps += 1

            interval = getattr(args, 'profile_interval', 0)
            if interval and (i % max(1, interval) == 0) and i > 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                it_s = i / max(1e-9, (time.time() - pbar.start_t)) if hasattr(pbar, 'start_t') else 0.0
                pbar.set_postfix(loss=f'{(total_loss/steps):.4f}', it_s=f'{it_s:.2f}')

        avg = total_loss / max(1, steps)
        writer.add_scalar('train/loss', avg, epoch)
        logger.info(f'[Train] epoch {epoch} | loss {avg:.6f}')
        return avg

    @torch.no_grad()
    def validate(epoch: int):
        net3d.eval(); ddpm3d.eval()
        total_loss = 0.0
        steps = 0
        for batch in tqdm(val_loader, desc=f'[Val] epoch {epoch}'):
            pc = batch['pointcloud'].to(device, non_blocking=True).float()
            V  = _make_voxels(pc)
            V  = V.to(device=device, dtype=torch.float32, non_blocking=True)
            V  = V.contiguous(memory_format=torch.channels_last_3d)

            B = V.size(0)
            t = torch.randint(1, args.T, (B,), device=device, dtype=torch.long)
            a_bar = ddpm3d.alphas_bar[t].view(B, 1, 1, 1, 1).to(dtype=V.dtype, device=V.device)
            eps   = torch.randn_like(V)
            x_t   = a_bar.sqrt() * V + (1.0 - a_bar).sqrt() * eps

            with autocast_ctx():
                loss = F.mse_loss(ddpm3d.predict_eps(x_t, t), eps)
            total_loss += loss.item()
            steps += 1

        avg = total_loss / max(1, steps)
        writer.add_scalar('val/loss', avg, epoch)
        logger.info(f'[Val] epoch {epoch} | loss {avg:.6f}')
        return avg

    # -------------------------------
    # Train loop
    # -------------------------------
    logger.info('Start training FDP 3D denoiser (optimized, clean resume)...')
    Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)

    for ep in range(start_epoch, args.epochs + 1):
        _ = train_one_epoch(ep)

        if (ep % args.val_freq == 0) or (ep == args.epochs):
            val_loss = validate(ep)
            ckpt = {
                'state_dict': ddpm3d.state_dict(),    # includes net + scheduler buffers
                'net_state_dict': net3d.state_dict(), # raw UNet3D weights
                'args': vars(args),
                'epoch': ep,
                'val_loss': val_loss,
            }
            # attach optimizer/scaler state for true resume
            try:
                ckpt['optimizer_state_dict'] = optimizer.state_dict()
            except Exception:
                ckpt['optimizer_state_dict'] = None
            try:
                ckpt['scaler_state_dict'] = scaler.state_dict() if (use_amp and args.amp_dtype == 'fp16') else None
            except Exception:
                ckpt['scaler_state_dict'] = None

            torch.save(ckpt, args.save_path)
            logger.info(f'[Save] saved to {args.save_path}')

            # Also keep a best snapshot via CheckpointManager (fallback if signature differs)
            try:
                ckpt_mgr.save(net3d, args, val_loss, step=ep)
            except TypeError:
                try:
                    ckpt_mgr.save(net3d, args, val_loss, ep)
                except TypeError:
                    best_path = os.path.join(os.path.dirname(args.save_path), f'denoise3d_best_ep{ep:04d}.pt')
                    torch.save(ckpt, best_path)
                    logger.info(f'[Save:FALLBACK] saved best to {best_path}')

    logger.info('Training finished.')


if __name__ == '__main__':
    main()
