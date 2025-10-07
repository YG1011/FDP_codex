import torch
from purifier.voxelizer import pointcloud_to_occupancy, occupancy_to_points

@torch.no_grad()
def roundtrip_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 构造一个单位球点云（1024 点）
    N, R = 1024, 128
    pts = torch.randn(N, 3, device=device)
    pts = pts / (pts.norm(dim=1, keepdim=True) + 1e-8)  # 单位球
    vol = pointcloud_to_occupancy(pts, res=R, splat_ks=1)  # (1,1,R,R,R)
    rec = occupancy_to_points(vol, num_points=N)          # (N,3)

    # 简单几何误差（非严格 Chamfer；仅作 smoke test）
    # 近似：原→重建（NN）误差
    # 这里为轻量起见，只计算 L2 平均距离的上界（随机子集）
    idx = torch.randperm(N, device=device)[:256]
    err = (pts[idx] - rec[idx]).pow(2).sum(dim=1).sqrt().mean().item()
    print(f'[RoundTrip] mean L2 ≈ {err:.4f} (res={R})')

if __name__ == '__main__':
    roundtrip_test()
