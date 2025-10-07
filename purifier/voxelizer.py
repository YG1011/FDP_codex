import torch
import torch.nn.functional as F
from typing import Tuple, Union

Tensor = torch.Tensor
Bounds = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


# -----------------------------
# 工具：坐标↔索引的归一化/反归一化
# -----------------------------
def _coords_to_indices(xyz: Tensor, res: int, bounds: Bounds) -> Tensor:
    """
    xyz: (..., 3), 坐标在 [xmin,xmax]×[ymin,ymax]×[zmin,zmax]
    return: (..., 3) in [0, res-1] 的浮点索引
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    idx = xyz.clone()
    idx[..., 0] = (idx[..., 0] - xmin) / (xmax - xmin) * (res - 1)
    idx[..., 1] = (idx[..., 1] - ymin) / (ymax - ymin) * (res - 1)
    idx[..., 2] = (idx[..., 2] - zmin) / (zmax - zmin) * (res - 1)
    return idx


def _indices_to_coords(idx: Tensor, res: int, bounds: Bounds) -> Tensor:
    """
    idx: (..., 3) in [0, res-1]
    return: (..., 3) 坐标回到给定 bounds，采样在体素中心（若 idx 为整数）
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    xyz = (idx + 0.5) / (res)  # 体素中心归一化到 [0,1]
    out = torch.empty_like(idx, dtype=torch.float32)
    out[..., 0] = xyz[..., 0] * (xmax - xmin) + xmin
    out[..., 1] = xyz[..., 1] * (ymax - ymin) + ymin
    out[..., 2] = xyz[..., 2] * (zmax - zmin) + zmin
    return out


# -----------------------------
# 点 → 体素（Occupancy / Density）
# -----------------------------
def pointcloud_to_occupancy(
    points: Tensor,
    res: int = 128,
    bounds: Bounds = ((-1, 1), (-1, 1), (-1, 1)),
    splat_ks: int = 1,
) -> Tensor:
    """
    将点云离散到体素密度体 (B,1,R,R,R)，采用 *三线性 splat*：
      每个点的“质量”按与 8 个相邻体素中心的距离权重分摊（保质心，更稳）。

    参数：
      - points: (N,3) 或 (B,N,3)，坐标建议已归一到 bounds（默认 [-1,1] 立方体）
      - res: 体素分辨率（>= 2）
      - splat_ks: 为兼容保留；>1 时建议在外层叠加 gaussian_smooth_occupancy 实现“厚涂”。

    返回： (B,1,R,R,R) float32，代表体素“密度/概率”，后续采样会按该权重进行。
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)

    B, N, _ = points.shape
    device = points.device
    dtype = torch.float32

    fidx = _coords_to_indices(points, res, bounds)
    fidx = fidx.clamp_(0.0, float(res - 1) - 1e-6)

    base = torch.floor(fidx).to(torch.long)
    frac = fidx - base.to(fidx.dtype)

    R = res
    vol_flat = torch.zeros((B, R * R * R), device=device, dtype=dtype)

    offs = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        device=device, dtype=torch.long,
    )

    wx0, wy0, wz0 = 1 - frac[..., 0], 1 - frac[..., 1], 1 - frac[..., 2]
    wx1, wy1, wz1 = frac[..., 0], frac[..., 1], frac[..., 2]
    weights8 = torch.stack([
        wx0 * wy0 * wz0, wx0 * wy0 * wz1, wx0 * wy1 * wz0, wx0 * wy1 * wz1,
        wx1 * wy0 * wz0, wx1 * wy0 * wz1, wx1 * wy1 * wz0, wx1 * wy1 * wz1,
    ], dim=2).to(dtype)  # (B,N,8)

    corners = base.unsqueeze(2) + offs.view(1, 1, 8, 3)
    corners = corners.clamp_(0, R - 1)

    lin = (
        corners[..., 0] * (R * R)
        + corners[..., 1] * R
        + corners[..., 2]
    ).view(B, -1)

    vol_flat.scatter_add_(1, lin, weights8.view(B, -1))
    return vol_flat.view(B, 1, R, R, R)



# -----------------------------
# 体素 → 点（按权重采样 + 体素内抖动 + 温度）
# -----------------------------
def occupancy_to_points(
    vol: Tensor,
    num_points: int,
    bounds: Bounds = ((-1, 1), (-1, 1), (-1, 1)),
    replace: bool = True,
) -> Tensor:
    """
    从体素密度体采样点云；按体素权重抽样，并在体素内做 *亚体素均匀抖动*，恢复连续坐标。
    额外加入温度 γ>1（强调高占用体素），以及抖动幅度缩放（jitter_scale）。

    兼容输入形状：(B,1,R,R,R) / (1,1,R,R,R)，返回 (B,N,3) 或 (N,3)。
    """
    if vol.dim() == 4:
        vol = vol.unsqueeze(0)  # (1,1,R,R,R)
    assert vol.dim() == 5 and vol.size(1) == 1, f"Expect (B,1,R,R,R), got {vol.shape}"

    B, _, R, _, _ = vol.shape
    device = vol.device
    out = []

    # ---- 温度与抖动设置（可按需调整） ----
    gamma = 2.2        # 采样温度：>1 更偏向高占用体素（1.5~2.0 通常效果好）
    jitter_scale = 0.15 # 抖动幅度：0.5 => 在 [-0.25, 0.25] 体素内抖动

    # 展平成权重并做温度变换；避免全 0 的情况
    w_flat = vol.reshape(B, -1).to(torch.float32)
    # 对全 0 行，用均匀分布替代以避免 NaN
    uniform = torch.full_like(w_flat, 1.0 / w_flat.size(1))
    has_mass = (w_flat.sum(dim=1, keepdim=True) > 0)
    w_base = torch.where(has_mass, w_flat, uniform)

    # 温度（幂）变换 + 归一化
    w_gamma = w_base.clamp(min=0).pow(gamma)
    w_norm = w_gamma / (w_gamma.sum(dim=1, keepdim=True) + 1e-12)

    for b in range(B):
        # Multinomial 按权重采样体素索引 - 添加安全检查
        try:
            choice = torch.multinomial(w_norm[b], num_samples=num_points, replacement=True)  # (N,)
        except RuntimeError as e:
            # 如果权重有问题，回退到均匀采样
            choice = torch.randint(0, R * R * R, (num_points,), device=device, dtype=torch.long)
        
        # 确保索引计算的安全性
        choice = choice.clamp(0, R * R * R - 1)  # 确保choice在有效范围内
        i = choice // (R * R)
        j = (choice // R) % R
        k = choice % R
        
        # 再次确保索引在有效范围内
        i = i.clamp(0, R - 1)
        j = j.clamp(0, R - 1) 
        k = k.clamp(0, R - 1)
        
        idx = torch.stack([i, j, k], dim=1).to(torch.float32)  # (N,3)

        # 体素内抖动：与 _indices_to_coords 中的 +0.5 结合，得到中心±抖动
        jitter = (torch.rand((num_points, 3), device=device, dtype=idx.dtype) - 0.5) * jitter_scale
        idx_frac = (idx + jitter).clamp(0.0, float(R - 1))
        coords = _indices_to_coords(idx_frac, R, bounds)  # (N,3)
        out.append(coords.unsqueeze(0))

    pts = torch.cat(out, dim=0)  # (B,N,3)
    if pts.size(0) == 1:
        pts = pts.squeeze(0)     # (N,3)
    return pts


# -----------------------------
# 可选：轻量“平滑”帮助频域稳定（替代 TSDF 的最简近似）
# -----------------------------
def gaussian_smooth_occupancy(vol: Tensor, sigma_vox: float = 1.0) -> Tensor:
    """
    对 occupancy 做 3D 高斯平滑（可作为简化 TSDF 的近似，便于频域稳定）。
    vol: (B,1,R,R,R) or (1,1,R,R,R)
    """
    if sigma_vox <= 0:
        return vol
    vol = vol.contiguous()  # 确保 NCDHW 连续
    # 为稳健起见核用 float32 计算后再转回原 dtype
    work_dtype = torch.float32
    vol_dtype = vol.dtype

    radius = max(1, int(2.0 * sigma_vox))
    coords = torch.arange(-radius, radius + 1, device=vol.device, dtype=work_dtype)
    kernel_1d = torch.exp(-0.5 * (coords / float(sigma_vox)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    kx = kernel_1d.view(1, 1, -1, 1, 1).to(vol_dtype)
    ky = kernel_1d.view(1, 1, 1, -1, 1).to(vol_dtype)
    kz = kernel_1d.view(1, 1, 1, 1, -1).to(vol_dtype)

    x = vol
    x = F.conv3d(x, kx, padding=(radius, 0, 0))
    x = F.conv3d(x, ky, padding=(0, radius, 0))
    x = F.conv3d(x, kz, padding=(0, 0, radius))

    # 归一到 [0,1]
    x_min = x.amin(dim=(2, 3, 4), keepdim=True)
    x_max = x.amax(dim=(2, 3, 4), keepdim=True)
    x = (x - x_min) / (x_max - x_min + 1e-8)
    return x
