#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import argparse
import csv
import importlib
from typing import Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 让脚本可从 project 根目录运行
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# ==== 你的工程内模块 ====
from models.classifiers.dgcnn import DGCNN_cls
from dataset.modelnet40 import ModelNet40
from purifier.FreqGuidedDiffuser3D import FreqGuidedDiffuser3D
from models.denoiser3d import DDPM3D, UNet3D
from purifier.voxelizer import pointcloud_to_occupancy, occupancy_to_points, gaussian_smooth_occupancy

# -----------------------------
# 小工具：安全加载 & 权重清洗
# -----------------------------
def safe_torch_load(path: str, map_location='cpu'):
    """优先使用 weights_only(更安全)，旧版 PyTorch 自动回退。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def extract_state_dict(obj):
    """兼容多种 ckpt 打包格式。"""
    for k in ['state_dict', 'model_state', 'model', 'net', 'net_state_dict', 'ema_state_dict']:
        if isinstance(obj, dict) and k in obj:
            return obj[k]
    return obj

def strip_prefix_if_present(state: dict, prefix: str) -> dict:
    """去掉常见前缀（如 'module.'、自定义网络名前缀等）。"""
    if not any(k.startswith(prefix) for k in state.keys()):
        return state
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

# -----------------------------
# 工具：形状/评测
# -----------------------------
def _ensure_b3n(pts: torch.Tensor) -> torch.Tensor:
    """
    支持 (B,N,3) 或 (B,3,N) -> 统一成 (B,3,N)
    """
    if pts.dim() != 3:
        raise ValueError(f'Expect 3D tensor, got {pts.shape}')
    if pts.size(1) == 3:
        return pts
    if pts.size(-1) == 3:
        return pts.transpose(1, 2).contiguous()
    raise ValueError(f'Unexpected shape {pts.shape}')

@torch.no_grad()
def evaluate(model, pts, labels):
    """
    评测单个 batch 的准确率
    pts : (B,N,3) 或 (B,3,N)
    labels: (B,) 或 (B,1)
    """
    pts = _ensure_b3n(pts)
    logits = model(pts)
    pred = logits.argmax(dim=1)

    labels = torch.as_tensor(labels, device=pred.device)
    if labels.dim() == 2 and labels.size(-1) == 1:
        labels = labels.squeeze(-1)
    else:
        labels = labels.view(-1)
    return (pred == labels).float().mean().item()

@torch.no_grad()
def chamfer_l2(p1, p2):
    # p1/p2: (B,N,3)
    # 返回批次平均的对称 Chamfer L2（简单实现，便于调试）
    d = torch.cdist(p1, p2)                # (B,N,N)
    return d.min(dim=-1).values.mean() + d.min(dim=-2).values.mean()

def renorm_unit_sphere(pc_bn3: torch.Tensor) -> torch.Tensor:
    """
    居中 + 单位球归一化 + 裁剪到[-1,1]
    支持 (B,N,3) 或 (N,3)
    """
    single = False
    if pc_bn3.dim() == 2:          # (N,3) -> (1,N,3)
        pc_bn3 = pc_bn3.unsqueeze(0)
        single = True

    pc = pc_bn3 - pc_bn3.mean(dim=1, keepdim=True)                 # 居中
    scale = pc.norm(dim=2).max(dim=1, keepdim=True).values.unsqueeze(-1) + 1e-6
    pc = (pc / scale).clamp(-1.0, 1.0)

    return pc.squeeze(0) if single else pc

# -----------------------------
# Attack 类解析（注册表优先，其次默认映射）
# -----------------------------
_DEFAULT_ATTACK_MAP = {
    'pgd_linf': ('pgd', 'PGD_Linf'),
    'pgdlinf': ('pgd', 'PGD_Linf'),
    'pgd': ('pgd', 'PGD_Linf'),

    'pgd_l2': ('pgdl2', 'PGD_L2'),
    'pgdl2': ('pgdl2', 'PGD_L2'),

    'cw': ('cw', 'CW'),
    'knn': ('knn', 'KNN'),

    'pointdrop': ('drop', 'PointDrop'),
    'drop': ('drop', 'PointDrop'),

    'pointadd': ('add', 'PointAdd'),
    'add': ('add', 'PointAdd'),

    'vanila': ('vanila', 'VANILA'),
    'vanilla': ('vanila', 'VANILA'),
}

def resolve_attack_class(name: Optional[str]):
    """
    返回 Attack 子类（类对象）。找不到则抛错。
    """
    if not name:
        raise ValueError('attack name is empty')
    key = name.lower()
    # 1) 尝试 registry
    try:
        attacks_pkg = importlib.import_module('attacks')
        if hasattr(attacks_pkg, 'ATTACK_REGISTRY'):
            reg = attacks_pkg.ATTACK_REGISTRY
            if key in reg:
                return reg[key]
    except Exception:
        pass
    # 2) 使用默认映射表
    if key in _DEFAULT_ATTACK_MAP:
        mod_name, cls_name = _DEFAULT_ATTACK_MAP[key]
        mod = importlib.import_module(f'attacks.{mod_name}')
        if hasattr(mod, cls_name):
            return getattr(mod, cls_name)
        raise ImportError(f'Class {cls_name} not found in module attacks.{mod_name}')
    raise ImportError(f'Attack "{name}" not found (no registry entry and no default mapping).')

# -----------------------------
# 主流程
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate purification (FreqGuidedDiffuser3D+DDPM3D) on ModelNet40 + DGCNN (pretrained download; Attack-class)'
    )
    # 数据/模型
    p.add_argument('--num-points', type=int, default=1024)
    p.add_argument('--scale-mode', type=str, default='none',
                   choices=['none','global_unit','shape_unit','shape_bbox','shape_half','shape_34','unit_sphere'])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=2)

    # 攻击
    p.add_argument('--attack', type=str, default='vanila',
                   help='attack key (e.g., pgd_linf, pgdl2, cw, knn, drop, add, vanila)')
    p.add_argument('--attack-seed', type=int, default=None)
    p.add_argument('--save-attack', action='store_true',
                   help='use Attack.save(...) to dump attacked dataset')
    p.add_argument('--attack-root', type=str, default=str(ROOT / 'data_attacked'))
    # attack params (PGD_Linf uses eps/alpha/steps/random_start)
    p.add_argument('--eps', type=float, default=0.05)
    p.add_argument('--alpha', type=float, default=0.01)
    p.add_argument('--steps', type=int, default=50)
    # 原来是 --random-start（store_true, default=True），不可从命令行关闭；改为以下更可控
    p.add_argument('--no-random-start', action='store_true',
                   help='disable random start in PGD-like attacks')

    # --- 频域净化器 & 扩散去噪 ---
    p.add_argument('--res', type=int, default=96)                 # 如显存吃紧，建议先 64/96 验证
    p.add_argument('--t-trunc', type=int, default=10)             # 逆扩散最大全步数（可被 n-steps 截断）
    p.add_argument('--DA-pix', type=int, default=12)               # 低频幅度半径(像素)
    p.add_argument('--DP-pix', type=int, default=6)               # 低频相位半径(像素)
    p.add_argument('--phase-delta', type=float, default=0.35)       # 相位带宽
    p.add_argument('--pre-smooth-sigma', type=float, default=0.8)  # 体素预平滑（轻度）
    p.add_argument('--splat-ks', type=int, default=1)              # 点→体素 splat 核大小
    p.add_argument('--no-channel', action='store_true',            # occupancy 是否带 channel 维
                   help='set to disable channel dim in occupancy')
    p.add_argument('--solver', type=str, default='ddim', choices=['ddim', 'ddpm'],
                   help='选择逆扩散求解器: ddim(默认, 可跳步) 或 ddpm(原始随机, 仅相邻步)')
    p.add_argument('--ddim-eta', type=float, default=0.0,
                   help='DDIM 方差系数 (eta=0 为确定性, >0 注入噪声)')

    # 模型权重路径（新增 DGCNN；DDPM 原有）
    p.add_argument('--dgcnn-ckpt', type=str, default='/home/linux/Documents/FDP/models/classifiers/pretrained/dgcnn_weights/model.1024.t7',
                   help='path to pretrained DGCNN classifier weights (e.g., model.1024.t7)')
    p.add_argument('--ddpm-ckpt', type=str, default='/home/linux/Documents/FDP/models/pretrained/45epoch_denoise3d.pt',
                   help='path to DDPM3D checkpoint (e.g., denoiser3d.pt)')

    # 其它控制
    p.add_argument('--mix-alpha', type=float, default=0.6,
                   help='低频幅度向输入A_adv的权重:越大越保守(默认0.8)')
    p.add_argument('--n-steps', type=int, default=10,
                   help='实际逆扩散步数; 0=bypass, <0=用 t_trunc, >0=固定步数')
    p.add_argument('--init-from-noise', action='store_true',
                   help='用N(0,1)初始化x_t;否则从 q_sample(x0,t_trunc) 初始化（更保真）')
    p.add_argument('--amp', action='store_true',
                   help='在净化阶段启用 autocast(float16) 与半精度 denoiser 以省显存')

    # 设备与输出
    p.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--csv', type=str, default=str(ROOT / 'results' / 'eval_purify.csv'))

    # diffusion
    p.add_argument('--T', type=int, default=1000,
                   help='DDPM total diffusion steps (must match training)')
    p.add_argument('--beta_1', type=float, default=1e-4,
                   help='beta schedule start (must match training)')
    p.add_argument('--beta_T', type=float, default=2e-2,
                   help='beta schedule end (must match training)')
    return p.parse_args()

    

def main():
    args = parse_args()
    device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    random_start = not args.no_random_start  # ← 统一用这个布尔值

    # --- 分类器 ---
    class Cfg:
        k = 20; emb_dims = 1024; dropout = 0.5
    model = DGCNN_cls(Cfg(), output_channels=40).to(device).eval()
    print('[model] DGCNN_cls initialized.')

    # 如果提供了本地 DGCNN 权重，就直接加载，避免依赖自动下载/gdown
    if args.dgcnn_ckpt:
        cls_raw = safe_torch_load(args.dgcnn_ckpt, map_location=device)
        cls_sd  = extract_state_dict(cls_raw)
        # 常见前缀清洗
        for pref in ['module.', 'dgcnn.']:
            cls_sd = strip_prefix_if_present(cls_sd, pref)
        try:
            missing, unexpected = model.load_state_dict(cls_sd, strict=True)
        except RuntimeError:
            # 仍报错就放宽
            missing, unexpected = model.load_state_dict(cls_sd, strict=False)
        if missing or unexpected:
            print(f'[WARN][DGCNN] missing={missing}, unexpected={unexpected}')
        print(f'[DGCNN] loaded: {args.dgcnn_ckpt}')
    else:
        print('[DGCNN] no --dgcnn-ckpt provided; using the model default behavior.')

    # --- 数据（注意本数据集构造签名：num_points, scale_mode, partition） ---
    test_dataset = ModelNet40(num_points=args.num_points, scale_mode=args.scale_mode, partition='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device=='cuda'),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )

    # --- 构造 DDPM 去噪器（与训练配置一致） ---
    # 注意：如果你训练时 UNet3D 的签名不同，请据实修改
    net3d = UNet3D(in_ch=1, base_ch=24, ch_mults=(1,2,4,8), time_dim=256, groups=8)
    denoiser = DDPM3D(net=net3d, T=args.T, beta_1=args.beta_1, beta_T=args.beta_T).to(device).eval()
    if args.amp and device == 'cuda':
        denoiser.half()  # 半精度模型以省显存

    if args.ddpm_ckpt:
        ckpt = safe_torch_load(args.ddpm_ckpt, map_location=device)

        # 1) DDPM 全量权重
        ddpm_state = extract_state_dict(ckpt)
        for pref in ('module.', 'ddpm.', 'model.'):
            ddpm_state = strip_prefix_if_present(ddpm_state, pref)
        missing, unexpected = denoiser.load_state_dict(ddpm_state, strict=True)
        if missing or unexpected:
            print(f'[WARN][DDPM] missing={missing}, unexpected={unexpected}')

        # 2) UNet 额外权重
        net_state = ckpt.get('net_state_dict')
        if net_state is not None:
            for pref in ('module.', 'net.'):
                net_state = strip_prefix_if_present(net_state, pref)
            net_missing, net_unexpected = net3d.load_state_dict(net_state, strict=True)
            if net_missing or net_unexpected:
                print(f'[WARN][UNet] missing={net_missing}, unexpected={net_unexpected}')

        # ---- 结构条目 & 数值 sanity check ----
        model_keys = set(denoiser.state_dict().keys())
        ckpt_keys  = set(ddpm_state.keys())
        extra_in_model = sorted(model_keys - ckpt_keys)
        extra_in_ckpt  = sorted(ckpt_keys - model_keys)
        print(f"[DDPM] 参数条目: 模型={len(model_keys)}, 权重={len(ckpt_keys)}")
        if extra_in_model or extra_in_ckpt:
            print(f"[DDPM] 差集: 模型缺少={extra_in_ckpt[:5]} | 模型多余={extra_in_model[:5]}")

        # 简单的权重范数检查
        print("alphas[:5] =", denoiser.alphas[:5])
        print("in_conv.weight norm =", net3d.in_conv.weight.norm())

        print(f'[DDPM] loaded: {args.ddpm_ckpt}')
    else:
        print('[DDPM] no --ddpm-ckpt provided; using randomly initialized denoiser (for debugging).')


    # --- 频域净化器（保守配置 + 新签名） ---
    pur_n_steps = None if args.n_steps < 0 else args.n_steps
    if args.solver == 'ddpm' and pur_n_steps not in (None, 0):
        print('[WARN] DDPM solver requires sequential steps; ignoring --n-steps and using full schedule.')
        pur_n_steps = None

    purifier = FreqGuidedDiffuser3D(
        denoise3d=denoiser,
        res=args.res,
        t_trunc=args.t_trunc,
        DA_pix=args.DA_pix,
        DP_pix=args.DP_pix,
        phase_delta=args.phase_delta,
        pre_smooth_sigma=args.pre_smooth_sigma,
        bounds=((-1,1),(-1,1),(-1,1)),
        splat_ks=args.splat_ks,
        use_channel=(not args.no_channel),
        mix_alpha=args.mix_alpha,
        n_steps=pur_n_steps,
        init_from_base=(not args.init_from_noise),
        ddim_eta=args.ddim_eta,
        solver=args.solver,
    ).to(device).eval()

    # AMP 上下文（仅净化阶段启用）
    amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if (args.amp and device=='cuda') else nullcontext()

    # ---- 解析/实例化 Attack 类 ----
    AttackClass = resolve_attack_class(args.attack)
    attack_obj = None

    # 1) 常见 PGD_Linf 签名 (model, device, eps, alpha, steps, random_start, seed)
    try:
        attack_obj = AttackClass(
            model,
            device,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            random_start=random_start,
            seed=args.attack_seed,
        )
        print(f'[ATTACK] Instantiated via (model, device, eps, alpha, steps, random_start, seed): {AttackClass}')
    except TypeError:
        attack_obj = None

    # 2) 回退：最小签名 (model, device, seed)
    if attack_obj is None:
        try:
            attack_obj = AttackClass(model, device, args.attack_seed)
            print(f'[ATTACK] Instantiated via (model, device, seed): {AttackClass}')
        except TypeError:
            attack_obj = None

    # 3) 最后回退
    if attack_obj is None:
        try:
            attack_obj = AttackClass(args.attack, model, device, args.attack_seed)
            print(f'[ATTACK] Instantiated via (name, model, device, seed): {AttackClass}')
        except Exception as e:
            raise RuntimeError(f'Instantiate Attack class failed (tried several signatures): {e}')

    # 兜底注入常用超参（若类 __init__ 没接收它们）
    def _ensure_attr(obj, name, val):
        if (not hasattr(obj, name)) or (getattr(obj, name) is None):
            setattr(obj, name, val)
    _ensure_attr(attack_obj, 'eps', args.eps)
    _ensure_attr(attack_obj, 'alpha', args.alpha)
    _ensure_attr(attack_obj, 'steps', args.steps)
    _ensure_attr(attack_obj, 'random_start', random_start)
    _ensure_attr(attack_obj, 'targeted', False)

    print(f'[ATTACK] Using Attack class: {attack_obj.__class__.__name__} (key="{args.attack}")')

    # --- 统计 ---
    acc_clean = acc_pur = acc_adv = 0.0
    cd_pur = 0.0
    n_batch = 0

    # 如果需要保存攻击数据，直接用 test_loader（其批次已是 dict 结构，满足 Attack.save 预期）
    if args.save_attack:
        try:
            print(f'[ATTACK:SAVE] Saving attacked dataset to {args.attack_root} via attack.save() ...')
            attack_obj.save(
                test_loader,
                root=args.attack_root,
                file_name=None,
                args={"attack": args.attack, "eps": args.eps, "alpha": args.alpha, "steps": args.steps}
            )
            print('[ATTACK:SAVE] Done.')
        except Exception as e:
            print(f'[ATTACK:SAVE] attack.save(...) failed: {e}')

    # ---- 评测循环 ----
    for batch in test_loader:
        pts = batch['pointcloud'].to(device).float()   # (B,N,3)
        labels = batch['cate'].to(device).long()       # (B,) or (B,1)

        # 1) CLEAN
        acc_clean += evaluate(model, pts, labels)

        # 2) ADV
        try:
            adv = attack_obj.attack(data=pts, labels=labels)
        except TypeError:
            adv = attack_obj.attack(pts, labels)
        adv = adv.clamp(-1.0, 1.0).to(device)
        acc_adv += evaluate(model, adv, labels)

        # （可选）体素回环误差，抽样 1024 点以提速
        with torch.no_grad():
            V = pointcloud_to_occupancy(pts, res=args.res, bounds=((-1,1),(-1,1),(-1,1)), splat_ks=args.splat_ks)
            P_rec = occupancy_to_points(V, num_points=pts.shape[1], bounds=((-1,1),(-1,1),(-1,1)), replace=True)
            P = min(1024, pts.shape[1]); idx = torch.randperm(pts.shape[1], device=pts.device)[:P]
            cd_voxel = chamfer_l2(pts[:, idx, :], P_rec[:, idx, :])
            print(f"[DBG] voxel_roundtrip_cd={cd_voxel:.6f}")

        # 3) PUR on ADV —— 一次性批量净化（FreqGuidedDiffuser3D 支持 (B,N,3)）
        with torch.inference_mode(), amp_ctx:
            pur_pts = purifier(adv, num_points=pts.shape[1])     # (B,N,3)

        pur_pts = renorm_unit_sphere(pur_pts)
        acc_pur += evaluate(model, pur_pts, labels)

        # Chamfer 也改为抽样 1024 点（原来对全 N 点，O(N^2) 可能很慢）
        P = min(1024, pts.shape[1])
        idx = torch.randperm(pts.shape[1], device=pts.device)[:P]
        cd_pur  += chamfer_l2(pts[:, idx, :], pur_pts[:, idx, :])

        # （可选）对干净样本也做一次净化，监控净化对 clean 的影响
        with torch.inference_mode(), amp_ctx:
            pur_clean = purifier(pts, num_points=pts.shape[1])
            pur_clean = renorm_unit_sphere(pur_clean)

        acc_pur_on_clean = evaluate(model, pur_clean, labels)

        # 轻量日志（随机子采样 1024 点计算 Chamfer）
        cd_clean = chamfer_l2(pts[:, idx, :],      pur_clean[:, idx, :])
        cd_adv   = chamfer_l2(adv[:, idx, :],      pur_pts[:, idx, :])
        improved_ratio = (cd_adv > cd_clean).float().mean().item()
        print(f"[DEBUG] pur_on_clean_acc={acc_pur_on_clean:.3f}  cd_clean={cd_clean:.6f}  cd_adv={cd_adv:.6f}  mix_alpha={getattr(purifier,'mix_alpha',None)}  improved_ratio={improved_ratio:.2f}  n_steps={getattr(purifier,'n_steps',None)}")

        n_batch += 1

    # --- 汇总 ---
    acc_clean /= n_batch
    acc_adv   /= n_batch
    acc_pur   /= n_batch
    cd_pur    /= n_batch

    print(f'[CLEAN] Acc = {acc_clean:.4f}')
    print(f'[ADV]   Acc = {acc_adv:.4f}')
    print(f'[PUR]   Acc = {acc_pur:.4f}, Chamfer(1024-sample) = {cd_pur:.6f}')

    # 保存 CSV
    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['CLEAN_Acc','ADV_Acc','PUR_Acc','Chamfer(1024)'])
        w.writerow([acc_clean, acc_adv, acc_pur, cd_pur])
    print(f'[SAVE] metrics -> {out_csv}')

if __name__ == '__main__':
    main()
