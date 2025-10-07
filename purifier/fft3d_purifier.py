import torch
import torch.nn as nn
from torch.fft import fftn, ifftn, fftshift, ifftshift

from .voxelizer import (
    pointcloud_to_occupancy,
    occupancy_to_points,
    gaussian_smooth_occupancy,
)

Tensor = torch.Tensor


def _lowpass_mask_3d(shape, radius: int, device=None, dtype=torch.float32) -> Tensor:
    _, _, R, _, _ = shape
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(R, device=device),
            torch.arange(R, device=device),
            torch.arange(R, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).to(dtype=dtype)
    center = (R - 1) / 2.0
    dist = torch.sqrt(((grid - center) ** 2).sum(dim=-1))
    mask = (dist <= float(radius)).to(dtype=dtype)
    return mask[None, None]


def _renorm_unit_sphere(pc_bn3: Tensor) -> Tensor:
    """居中 + 单位球归一化 + clip，支持(N,3)/(B,N,3)。"""
    single = False
    if pc_bn3.dim() == 2:
        pc_bn3 = pc_bn3.unsqueeze(0)
        single = True
    pc = pc_bn3 - pc_bn3.mean(dim=1, keepdim=True)
    scale = pc.norm(dim=2).max(dim=1, keepdim=True).values.unsqueeze(-1) + 1e-6  # (B,1,1)
    pc = (pc / scale).clamp(-1.0, 1.0)
    return pc.squeeze(0) if single else pc


class FFT3DPurifier(nn.Module):
    """
    3D 频域净化器（低频幅度软混合 ASE）。
    流程：点→体素 → 3D-FFT(零频居中) → 低频幅度软混合 → 逆移&IFFT → 体素→点 → 归一化
    """

    def __init__(
        self,
        DA: int = 3,               # 更保守的低频半径
        mix_alpha: float = 0.2,    # 更保守的混合系数
        res: int = 256,            # 更高分辨率，减小量化误差
        pre_smooth_sigma: float = 0.0,  # 0=关闭；若>0走高斯
        splat_ks: int = 1,
        bounds=((-1, 1), (-1, 1), (-1, 1)),
        passthrough_voxel_only: bool = False,  # 调试开关：只做 点→体素→点
    ):
        super().__init__()
        assert 0.0 <= mix_alpha <= 1.0
        self.DA = int(DA)
        self.mix_alpha = float(mix_alpha)
        self.res = int(res)
        self.pre_smooth_sigma = float(pre_smooth_sigma)
        self.splat_ks = int(splat_ks)
        self.bounds = bounds
        self.passthrough_voxel_only = bool(passthrough_voxel_only)

    @torch.no_grad()
    def forward(self, adv_points: Tensor) -> Tensor:
        assert adv_points.dim() == 2 and adv_points.size(-1) == 3, f"Expect (N,3), got {adv_points.shape}"

        # 1) 点→体素
        V_adv = pointcloud_to_occupancy(
            adv_points, res=self.res, bounds=self.bounds, splat_ks=self.splat_ks
        )  # (1,1,R,R,R)

        # --- Debug 模式：只测试 点↔体素 闭环是否稳定 ---
        if self.passthrough_voxel_only:
            pts = occupancy_to_points(
                V_adv, num_points=adv_points.size(0), bounds=self.bounds, replace=True
            )
            return _renorm_unit_sphere(pts)

        # 2) 基准体素：默认不再 avgpool，避免平滑引入偏差
        if self.pre_smooth_sigma > 0:
            V_base = gaussian_smooth_occupancy(V_adv, sigma_vox=self.pre_smooth_sigma)
        else:
            V_base = V_adv

        # 3) 频域：零频居中
        dims = (-3, -2, -1)
        F0   = fftshift(fftn(V_base, dim=dims), dim=dims)
        Fadv = fftshift(fftn(V_adv,  dim=dims), dim=dims)

        A0, P0  = torch.abs(F0), torch.angle(F0)
        Aadv,Padv = torch.abs(Fadv), torch.angle(Fadv)

        # 半径保护
        R = V_adv.shape[-1]
        radius = int(max(1, min(self.DA, R // 2)))

        # 4) 低频幅度软混合（低频：α*Aadv+(1-α)*A0；高频：A0）
        mask = _lowpass_mask_3d(A0.shape, radius=radius, device=A0.device, dtype=A0.dtype)
        A_mix_low = self.mix_alpha * Aadv + (1.0 - self.mix_alpha) * A0
        A_new = Aadv * (1.0 - mask) + A_mix_low * mask

        # 5) 逆移&IFFT
        P_new = Padv
        F_new = ifftshift(A_new * torch.exp(1j * P_new), dim=dims)
        V_new = ifftn(F_new, dim=dims).real

        # 6) 体素→点
        purified = occupancy_to_points(
            V_new, num_points=adv_points.size(0), bounds=self.bounds, replace=True
        )

        # 7) 回到训练分布
        purified = _renorm_unit_sphere(purified)
        return purified
