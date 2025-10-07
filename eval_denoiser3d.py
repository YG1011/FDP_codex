# eval_denoiser.py
import os
import torch
import torch.nn as nn
from argparse import Namespace
from train_denoiser import (
    ModelNet40, DataLoader, UNet3D, DDPM3D,
    pointcloud_to_occupancy, gaussian_smooth_occupancy
)

# =========================
# 0) 后端 & 同步设置
# =========================
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # 强制同步便于拿到真实栈
USE_CUDNN = False  # ← 如需测试 cuDNN 性能可改成 True；遇错会自动回退
torch.backends.cudnn.enabled = USE_CUDNN
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("high")

# =========================
# 1) 评估参数（需与训练一致）
# =========================
args = Namespace(
    res=96, T=1000, beta_1=1e-4, beta_T=2e-2,
    base_ch=24, num_points=1024, train_batch_size=1,
    val_batch_size=1, num_workers=2,          # 定位阶段先用 2
    pre_smooth_sigma=0.8, splat_ks=1,
    amp=False
)

# =========================
# 2) 设备
# =========================
if torch.cuda.is_available():
    gpu_idx = 0
    torch.cuda.set_device(gpu_idx)
    device = torch.device(f"cuda:{gpu_idx}")
else:
    device = torch.device("cpu")

# =========================
# 3) DataLoader
# =========================
val_set = ModelNet40(partition='test', scale_mode='none', num_points=args.num_points)
val_loader = DataLoader(
    val_set, batch_size=args.val_batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=(device.type == "cuda")
)

# =========================
# 4) 构建模型（结构必须与 ckpt 一致）
# =========================
net = UNet3D(in_ch=1, base_ch=args.base_ch, time_dim=256)
ddpm = DDPM3D(net, T=args.T, beta_1=args.beta_1, beta_T=args.beta_T)

# =========================
# 5) 安全加载 ckpt（严格匹配）
# =========================
ckpt_path = 'models/pretrained/45epoch_denoise3d.pt'
def safe_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')

ckpt = safe_load(ckpt_path)
state = ckpt
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    state = ckpt['state_dict']

missing, unexpected = ddpm.load_state_dict(state, strict=True)  # 避免“半加载”
# 如你的 ckpt 是分开存的 net/ddpm 两份，请改成：
# net.load_state_dict(ckpt['net_state_dict'], strict=True)
# ddpm.load_state_dict(ckpt['ddpm_state_dict'], strict=True)

net.to(device).eval()
ddpm.to(device).eval()

assert hasattr(ddpm, 'alphas_bar') and ddpm.alphas_bar.numel() >= args.T, \
    f'Expected alphas_bar length >= {args.T}, got {ddpm.alphas_bar.numel()}'

# =========================
# 6) 调试 hook（含 ConvTranspose3d 形状打印）
# =========================
def attach_debug_hooks(model: nn.Module):
    watch_types = (nn.Conv3d, nn.GroupNorm, nn.BatchNorm3d,
                   nn.SiLU, nn.ReLU, nn.Upsample, nn.ConvTranspose3d)
    for name, m in model.named_modules():
        if isinstance(m, watch_types):
            def _make_hook(layer_name, is_deconv):
                def _hook(mod, inp, out):
                    def _shape(x):
                        return tuple(x.shape) if torch.is_tensor(x) else type(x).__name__
                    xs = out if isinstance(out, (tuple, list)) else [out]
                    if is_deconv:
                        w = getattr(mod, "weight", None)
                        wshape = tuple(w.shape) if torch.is_tensor(w) else None
                        ishapes = [ _shape(i) for i in (inp if isinstance(inp, (tuple,list)) else [inp]) ]
                        oshape  = _shape(xs[0])
                        print(f"[Deconv] {layer_name} | W={wshape} | IN={ishapes} | OUT={oshape}")
                    # NaN/Inf 检查
                    for x in xs:
                        if torch.is_tensor(x) and not torch.isfinite(x).all():
                            raise RuntimeError(f"[NaN/Inf] after {layer_name} | shape={tuple(x.shape)}")
                return _hook
            m.register_forward_hook(_make_hook(name, isinstance(m, nn.ConvTranspose3d)))

attach_debug_hooks(net)

# =========================
# 7) 基础数值检查
# =========================
def _check(x, name):
    assert torch.isfinite(x).all(), f"{name} contains NaN/Inf, shape={tuple(x.shape)}"
    return x

# =========================
# 8) 体素化：确保 (B,1,D,H,W)
# =========================
def make_voxels(pc):
    # 将点坐标裁剪到边界，避免落桶越界
    pc = pc.clamp_(-1.0, 1.0)
    v = pointcloud_to_occupancy(pc, res=args.res,
                                bounds=((-1, 1),) * 3, splat_ks=args.splat_ks)
    v = _check(v, "occupancy").clamp_(0, 1).to(dtype=torch.float32).contiguous()
    if v.dim() == 4:
        v = v.unsqueeze(1)  # (B,1,D,H,W)
    elif v.dim() != 5:
        raise RuntimeError(f"Unexpected voxel dim: {v.dim()}, expected 5D (B,1,D,H,W)")
    if args.pre_smooth_sigma and args.pre_smooth_sigma > 0:
        v = gaussian_smooth_occupancy(v, sigma_vox=args.pre_smooth_sigma)
        v = _check(v, "smoothed_occupancy").clamp_(0, 1).contiguous()
    return v

# =========================
# 9) 随机输入“自检”模型（结构/权重是否能跑通）
# =========================
with torch.no_grad():
    B = 1
    rand_x = torch.randn(B, 1, args.res, args.res, args.res, device=device, dtype=torch.float32)
    t0 = torch.randint(0, args.T, (B,), device=device, dtype=torch.long)
    _ = ddpm.predict_eps(rand_x, t0)  # 若这里报错 → 模型/权重/结构问题

# =========================
# 10) 评估主循环（含 cuDNN 回退）
# =========================
loss_sum = 0.0
steps = 0

def predict_eps_safe(x_t, t):
    """开启 cuDNN 时若遇到 CUDNN_STATUS_EXECUTION_FAILED，自动回退并重试一次。"""
    try:
        return ddpm.predict_eps(x_t, t)
    except RuntimeError as e:
        msg = str(e)
        if "CUDNN_STATUS_EXECUTION_FAILED" in msg and torch.backends.cudnn.enabled:
            print("[WARN] cuDNN execution failed in ConvTranspose3d, falling back to native kernels...")
            torch.backends.cudnn.enabled = False
            out = ddpm.predict_eps(x_t, t)  # 重试
            print("[INFO] Fallback succeeded.")
            return out
        raise

with torch.no_grad():
    for step, batch in enumerate(val_loader):
        pc = batch['pointcloud'].float()
        _check(pc, "pointcloud_cpu")
        pc = pc.to(device, non_blocking=True)

        V = make_voxels(pc).to(device, non_blocking=True)
        _check(V, "V_on_device")

        B = V.size(0)
        t = torch.randint(0, args.T, (B,), device=device, dtype=torch.long)
        a_bar = ddpm.alphas_bar.index_select(0, t).view(B, 1, 1, 1, 1).to(V.dtype)
        _check(a_bar, "a_bar")

        eps = torch.randn_like(V)
        x_t = a_bar.sqrt() * V + (1.0 - a_bar).sqrt() * eps
        _check(x_t, "x_t")

        # 关键：这里若 cuDNN 报错会自动回退一次
        eps_pred = predict_eps_safe(x_t, t)
        _check(eps_pred, "eps_pred")

        loss = torch.nn.functional.mse_loss(eps_pred, eps)
        _check(loss, "loss_tensor")

        loss_sum += loss.item()
        steps += 1

print(f'Validation loss: {loss_sum / max(1, steps):.6f}')
