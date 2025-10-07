# models/denoiser3d.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= 安全封装与工具 =========
class SafeConv3d(nn.Conv3d):
    def forward(self, input):
        x = input.contiguous()
        if torch.any(~torch.isfinite(x)):
            raise RuntimeError(f"SafeConv3d input has NaN/Inf: shape={tuple(x.shape)}")
        return super().forward(x)

class SafeDeconv3d(nn.ConvTranspose3d):
    def forward(self, input, output_size=None):
        x = input.contiguous()
        if torch.any(~torch.isfinite(x)):
            raise RuntimeError(f"SafeDeconv3d input has NaN/Inf: shape={tuple(x.shape)}")
        return super().forward(x, output_size=output_size)

def make_groupnorm(num_channels: int, num_groups: int = 32, eps: float = 1e-5):
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    if g <= 0:
        g = 1
    return nn.GroupNorm(g, num_channels, eps=eps)

def _cl3d(y: torch.Tensor) -> torch.Tensor:
    # 统一 3D 连续内存格式（对 cuDNN 更友好）
    return y.contiguous(memory_format=torch.channels_last_3d) if y.is_cuda else y.contiguous()

def _match_spatial(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # 让 x 的 (D,H,W) 与 ref 一致；若不同，用最近邻插值对齐
    if x.shape[2:] != ref.shape[2:]:
        x = F.interpolate(x, size=ref.shape[2:], mode='nearest')
    return x

# ---------- 时间嵌入 ----------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) long/int tensor of timesteps
        return: (B, dim)
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / half)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:  # pad if odd
            emb = F.pad(emb, (0, 1))
        return emb

# ---------- ResBlock3D（FiLM 调制，安全卷积与自适配 GroupNorm） ----------
class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=32):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = make_groupnorm(in_ch, groups)
        self.act1  = nn.SiLU()
        self.conv1 = SafeConv3d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),  # scale, shift
        )

        self.norm2 = make_groupnorm(out_ch, groups)
        self.act2  = nn.SiLU()
        self.conv2 = SafeConv3d(out_ch, out_ch, 3, padding=1)

        self.skip = (in_ch == out_ch) and nn.Identity() or SafeConv3d(in_ch, out_ch, 1)

    def forward(self, x, t_emb):
        """
        x: (B,C,D,H,W)
        t_emb: (B, time_dim)
        """
        if torch.any(~torch.isfinite(x)):
            raise RuntimeError(f"ResBlock3D input NaN/Inf: {tuple(x.shape)}")

        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(_cl3d(h))

        # FiLM: scale-shift
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)  # (B,C), (B,C)
        scale = scale.view(scale.shape[0], scale.shape[1], 1, 1, 1)
        shift = shift.view(shift.shape[0], shift.shape[1], 1, 1, 1)

        h = self.norm2(h * (1 + scale) + shift)
        h = self.act2(h)
        h = self.conv2(_cl3d(h))

        x_res = self.skip(x)
        out = h + x_res
        if torch.any(~torch.isfinite(out)):
            raise RuntimeError(f"ResBlock3D output NaN/Inf: {tuple(out.shape)}")
        return _cl3d(out)

# ---------- 上/下采样（安全版） ----------
class Downsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = SafeConv3d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x):
        return _cl3d(self.op(_cl3d(x)))

class Upsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = SafeDeconv3d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x):
        return _cl3d(self.op(_cl3d(x)))

# ---------- U-Net 主干 ----------
class UNet3D(nn.Module):
    """
    适中规模、训练稳定的 3D U-Net
    """
    def __init__(self, in_ch=1, base_ch=24, ch_mults=(1,2,4,8), time_dim=256, groups=8):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim*4),
            nn.SiLU(),
            nn.Linear(time_dim*4, time_dim),
        )

        # 输入投影
        self.in_conv = SafeConv3d(in_ch, base_ch, 3, padding=1)

        # 下采样阶段
        downs = []
        ch = base_ch
        self.downs_out_ch = []
        for i, m in enumerate(ch_mults):
            out_ch = base_ch * m
            block1 = ResBlock3D(ch, out_ch, time_dim, groups)
            block2 = ResBlock3D(out_ch, out_ch, time_dim, groups)
            self.downs_out_ch.append(out_ch)
            if i != len(ch_mults) - 1:
                down = Downsample3D(out_ch)
            else:
                down = nn.Identity()
            downs.append(nn.ModuleList([block1, block2, down]))
            ch = out_ch
        self.downs = nn.ModuleList(downs)

        # bottleneck
        self.mid1 = ResBlock3D(ch, ch, time_dim, groups)
        self.mid2 = ResBlock3D(ch, ch, time_dim, groups)

        # 上采样阶段
        ups = []
        for i, m in enumerate(reversed(ch_mults)):
            skip_ch = base_ch * m
            block1 = ResBlock3D(ch + skip_ch, skip_ch, time_dim, groups)
            block2 = ResBlock3D(skip_ch,       skip_ch, time_dim, groups)
            if i != len(ch_mults) - 1:
                up = Upsample3D(skip_ch)
            else:
                up = nn.Identity()
            ups.append(nn.ModuleList([block1, block2, up]))
            ch = skip_ch
        self.ups = nn.ModuleList(ups)

        # 输出投影
        self.out = nn.Sequential(
            make_groupnorm(ch, groups),
            nn.SiLU(),
            SafeConv3d(ch, in_ch, 3, padding=1),
        )

    def forward(self, x, t):
        # 归一化 timestep 形状
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=torch.long).view(-1)
        if t.ndim == 0:
            t = t[None].expand(x.shape[0])
        t_emb = self.time_emb(t)

        # in
        h = self.in_conv(_cl3d(x))

        # down path
        skips = []
        for b1, b2, down in self.downs:
            h = b1(h, t_emb)
            h = b2(h, t_emb)
            skips.append(h)
            h = down(h)

        # mid
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # up path（concat 前做空间对齐）
        for (b1, b2, up), skip in zip(self.ups, reversed(skips)):
            skip = _match_spatial(skip, h)
            h = torch.cat([h, skip], dim=1)
            h = b1(h, t_emb)
            h = b2(h, t_emb)
            h = up(h)

        out = self.out(h)
        return out

# ---------- DDPM3D wrapper（保持你现有接口） ----------
class DDPM3D(nn.Module):
    """
    net(x_t, t) -> eps
    线性 beta 调度
    """
    def __init__(self, net: nn.Module, T: int = 1000, beta_1: float = 1e-4, beta_T: float = 2e-2):
        super().__init__()
        assert isinstance(net, nn.Module)
        self.net = net
        self.T = int(T)

        betas = torch.linspace(beta_1, beta_T, self.T, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_bar[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("alphas_bar_prev", alphas_bar_prev)

    def forward(self, x_t: torch.Tensor, t):
        return self.net(x_t, self._normalize_t(t, x_t))

    # 兼容旧名
    def predict_eps(self, x_t: torch.Tensor, t):
        return self.forward(x_t, t)

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, t, noise: torch.Tensor = None):
        t = self._normalize_t(t, x0)
        if noise is None: noise = torch.randn_like(x0)
        a_bar = self._gather(self.alphas_bar, t, x0)
        return a_bar.sqrt()*x0 + (1.0 - a_bar).sqrt()*noise

    def predict_x0(self, x_t: torch.Tensor, t, eps: torch.Tensor):
        t = self._normalize_t(t, x_t)
        a_bar = self._gather(self.alphas_bar, t, x_t)
        return (x_t - (1.0 - a_bar).sqrt()*eps) / a_bar.sqrt().clamp_min(1e-12)

    @torch.no_grad()
    def sample_step(self, x_t: torch.Tensor, t, z: torch.Tensor=None):
        t     = self._normalize_t(t, x_t)
        eps   = self.predict_eps(x_t, t)
        a_t   = self._gather(self.alphas, t, x_t)
        a_bar = self._gather(self.alphas_bar, t, x_t)
        beta  = self._gather(self.betas, t, x_t)

        mean = (1.0 / a_t.sqrt()) * (x_t - ((1.0 - a_t)/(1.0 - a_bar).sqrt().clamp_min(1e-12)) * eps)
        if z is None: z = torch.randn_like(x_t)
        sigma = beta.sqrt()
        mask  = (t > 0).float().view(-1, *([1]*(x_t.ndim-1)))
        return mean + mask * sigma * z

    # ---- utils ----
    def _normalize_t(self, t, x_like: torch.Tensor) -> torch.Tensor:
        device = x_like.device; B = x_like.shape[0]
        if isinstance(t, int):
            t = torch.full((B,), t, device=device, dtype=torch.long)
        elif isinstance(t, (list, tuple)):
            t = torch.tensor(t, device=device, dtype=torch.long)
        elif isinstance(t, torch.Tensor):
            t = t.to(device=device, dtype=torch.long).view(-1)
            if t.numel()==1 and B>1: t = t.expand(B)
        else:
            raise TypeError(f'bad t type: {type(t)}')
        assert t.numel()==B, f't size {t.numel()} != batch {B}'
        return t

    def _gather(self, vec: torch.Tensor, t: torch.Tensor, x_like: torch.Tensor) -> torch.Tensor:
        # 确保时间步索引在有效范围内，避免越界访问
        t_safe = t.clamp(0, self.T-1).to(torch.long)
        out = vec.gather(0, t_safe)
        return out.view(out.shape[0], *([1]*(x_like.ndim-1))).to(dtype=x_like.dtype)
