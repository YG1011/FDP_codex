# purifier/freq_guided_diffuser3d.py
import torch
import torch.nn as nn
from torch.fft import fftn, ifftn, fftshift, ifftshift
from typing import Optional, List, Tuple

from .voxelizer import (
    pointcloud_to_occupancy,
    occupancy_to_points,
    gaussian_smooth_occupancy,
)
from models.denoiser3d import DDPM3D


def _lowpass_mask(R: int, radius: int, device, dtype):
    g = torch.stack(
        torch.meshgrid(
            torch.arange(R, device=device),
            torch.arange(R, device=device),
            torch.arange(R, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).to(dtype)
    c = (R - 1) / 2.0
    dist = torch.sqrt(((g - c) ** 2).sum(-1))
    m = (dist <= float(radius)).to(dtype)
    return m[None, None, ...]


def _phase_project_to_band(phase_est, phase_ref, delta: float):
    # wrap to (-pi, pi], then clamp to +-delta
    diff = (phase_est - phase_ref + torch.pi) % (2 * torch.pi) - torch.pi
    diff = diff.clamp(-delta, delta)
    return phase_ref + diff


class FreqGuidedDiffuser3D(nn.Module):
    """
    频域引导的 3D 逆扩散（DDIM 版本）
      - 全程在 [0,1] 数值域
      - 低频幅度混合（向 V_base 靠）+ 低频相位投影（以 V_base 为带）
      - DDIM 跳步：使用 (t_cur, t_next) 成对推进；eta=0 确定性，>0 可注入少量噪声
      - 不传 n_steps 则走连续相邻步（仍用 DDIM 公式, t_next=t_cur-1）
    """

    def __init__(
        self,
        denoise3d: DDPM3D,
        res: int = 96,
        t_trunc: int = 1000,               # 回退的“高阶步数”上限；>=T 表示直到 0
        DA_pix: int = 8,
        DP_pix: Optional[int] = 6,
        phase_delta: float = 0.30,
        pre_smooth_sigma: float = 0.8,
        bounds=((-1, 1), (-1, 1), (-1, 1)),
        splat_ks: int = 1,
        use_channel: bool = True,
        mix_alpha: float = 0.75,           # 建议 0.6~0.85
        n_steps: Optional[int] = None,     # DDIM 跳步数；None=连续
        init_from_base: bool = True,
        ddim_eta: float = 0.0,             # DDIM 方差系数；0=确定性
    ):
        super().__init__()
        self.denoise3d = denoise3d
        self.res = int(res)
        self.t_trunc = int(t_trunc)
        self.DA = int(DA_pix)
        self.DP = int(DP_pix if DP_pix is not None else max(1, DA_pix // 2))
        self.phase_delta = float(phase_delta)
        self.pre_smooth_sigma = float(pre_smooth_sigma)
        self.bounds = bounds
        self.splat_ks = int(splat_ks)
        self.use_channel = bool(use_channel)

        self.mix_alpha = float(mix_alpha)
        self.n_steps = None if (n_steps is None) else int(n_steps)
        self.init_from_base = bool(init_from_base)
        self.ddim_eta = float(ddim_eta)

        # 掩模缓存
        self.register_buffer("_mask_A", None, persistent=False)
        self.register_buffer("_mask_P", None, persistent=False)

    # ---------- 时间表：返回成对 (t_cur, t_next) ----------
    def _schedule_pairs(self, T: int) -> List[Tuple[int, int]]:
        """
        若 n_steps is None: 连续相邻步 [(T-1, T-2), (T-2, T-3), ...]
        否则: DDIM 均匀跳步，从 T-1 均匀到 end_t（含两端），再配对相邻端点
        """
        end_t = max(0, T - int(self.t_trunc))

        if self.n_steps is None:
            seq = list(range(T - 1, end_t, -1))  # 不包含 end_t 本身，留给配对的 t_next
            return [(t, t - 1) for t in seq]     # (T-1,T-2),...,(end_t+1,end_t)

        # DDIM: 均匀取 n_steps 段 -> 需要 n_steps+1 个端点
        S = max(1, int(self.n_steps))
        grid = torch.linspace(T - 1, end_t, steps=S + 1)
        idx = torch.round(grid).to(torch.long).tolist()

        # 严格递减去重
        dedup = []
        for t in idx:
            if not dedup or t < dedup[-1]:
                dedup.append(int(t))
        if dedup[0] != T - 1:
            dedup[0] = T - 1
        if dedup[-1] != end_t:
            dedup[-1] = end_t
        if len(dedup) < 2:
            dedup = [T - 1, end_t]

        return [(dedup[i], dedup[i + 1]) for i in range(len(dedup) - 1)]

    def _masks(self, device, dtype):
        R = self.res
        need_rebuild = (
            self._mask_A is None
            or self._mask_A.device != device
            or self._mask_A.shape[-1] != R
        )
        if need_rebuild:
            mA = _lowpass_mask(R, self.DA, device, torch.float32).to(dtype)
            mP = _lowpass_mask(R, self.DP, device, torch.float32).to(dtype)
            self._mask_A = mA
            self._mask_P = mP
        else:
            if self._mask_A.dtype != dtype:
                self._mask_A = self._mask_A.to(dtype)
                self._mask_P = self._mask_P.to(dtype)
        return self._mask_A, self._mask_P

    # ---------- 主流程 ----------
    @torch.no_grad()
    def __call__(self, pc_adv: torch.Tensor, num_points: Optional[int] = None):
        """
        pc_adv: (B,N,3) in self.bounds
        return : (B,N,3)
        """
        B, N, _ = pc_adv.shape
        if num_points is None:
            num_points = N
        device, in_dtype = pc_adv.device, pc_adv.dtype

        # 旁路
        if (self.n_steps is not None) and (self.n_steps <= 0):
            return pc_adv

        # 1) 点 -> 体素 V∈[0,1]
        V_adv = pointcloud_to_occupancy(
            pc_adv, res=self.res, bounds=self.bounds, splat_ks=self.splat_ks
        )
        V_base = (
            gaussian_smooth_occupancy(V_adv, sigma_vox=self.pre_smooth_sigma)
            if self.pre_smooth_sigma > 0
            else V_adv
        )

        # 2) 时间表 + 初始 x_t
        T = int(getattr(self.denoise3d, "T", len(self.denoise3d.betas)))
        pairs = self._schedule_pairs(T)
        t_start = pairs[0][0]

        if self.init_from_base:
            x_t = self.denoise3d.q_sample(V_base, t=t_start)
        else:
            x_t = torch.randn_like(V_base)

        # 3) 频域锚（使用平滑后的 V_base 更稳）
        dims = (-3, -2, -1)
        real_fft_dtype = (
            torch.float32 if V_adv.dtype in (torch.float16, torch.bfloat16) else V_adv.dtype
        )
        F_base = fftshift(fftn(V_base.to(real_fft_dtype), dim=dims), dim=dims)
        A_base = torch.abs(F_base).to(V_base.dtype)
        P_base = torch.angle(F_base).to(V_base.dtype)
        MA, MP = self._masks(device, A_base.dtype)

        alphas_bar = self.denoise3d.alphas_bar.to(device)  # (T,)

        # 4) 按 (t_cur -> t_next) 执行 DDIM 更新
        for (t_cur, t_next) in pairs:
            a_cur = alphas_bar[t_cur].clamp_min(1e-12)
            a_nxt = alphas_bar[t_next].clamp_min(1e-12)
            sqrt_a_cur = torch.sqrt(a_cur)
            sqrt_one_minus_a_cur = torch.sqrt((1 - a_cur).clamp_min(1e-12))

            # 4.1 预测 eps 与 x0_est
            t_tensor = torch.full((B,), t_cur, device=device, dtype=torch.long)
            eps_pred = self.denoise3d(x_t, t_tensor)
            x0_est = (x_t - sqrt_one_minus_a_cur * eps_pred) / sqrt_a_cur

            # 4.2 对 x0_est 做频域约束 -> x0_hat
            F_est = fftshift(fftn(x0_est.to(real_fft_dtype), dim=dims), dim=dims)
            A_est = torch.abs(F_est).to(x0_est.dtype)
            P_est = torch.angle(F_est).to(x0_est.dtype)

            A_mix_low = self.mix_alpha * A_base + (1.0 - self.mix_alpha) * A_est
            A_new = A_est * (1.0 - MA) + A_mix_low * MA

            P_low = _phase_project_to_band(P_est, P_base, self.phase_delta)
            P_new = P_est * (1.0 - MP) + P_low * MP

            F_new = ifftshift(
                A_new.to(real_fft_dtype) * torch.exp(1j * P_new.to(real_fft_dtype)), dim=dims
            )
            x0_hat = ifftn(F_new, dim=dims).real.to(x0_est.dtype)

            # 4.3 DDIM 公式
            # eps_tilde 使用 x0_hat 重估（与 DDIM 推导一致）
            eps_tilde = (x_t - torch.sqrt(a_cur) * x0_hat) / torch.sqrt(
                (1 - a_cur).clamp_min(1e-12)
            )

            if self.ddim_eta == 0.0:
                # 确定性 DDIM
                x_next = torch.sqrt(a_nxt) * x0_hat + torch.sqrt(
                    (1 - a_nxt).clamp_min(1e-12)
                ) * eps_tilde
            else:
                # 随机 DDIM（可选）
                sigma = self.ddim_eta * torch.sqrt(
                    (1 - a_nxt) / (1 - a_cur) * (1 - a_cur / a_nxt)
                ).to(x0_hat.dtype)
                noise = torch.randn_like(x_t)
                c = torch.sqrt((1 - a_nxt - sigma**2).clamp_min(0.0))
                x_next = torch.sqrt(a_nxt) * x0_hat + c * eps_tilde + sigma * noise

            x_t = x_next

        V_new = x_t.clamp(0.0, 1.0)
        pc_out = occupancy_to_points(
            V_new, num_points=num_points, bounds=self.bounds, replace=True
        )
        return pc_out.to(in_dtype)
