import torch
import torch.nn.functional as F

def trilinear_sdf_at_points(
    sdf_d,                 # (V^3,) 或 (V, V, V) 的稠密SDF（顶点格）
    pcls,                  # (B,P,3) 或 (P,3) 点云（世界坐标）
    bbox_min, bbox_max,    # (3,) 世界坐标AABB
    res: int = 256,        # cell分辨率 = 256  -> 顶点格 V = 257
    eps: float = 1e-6,
):
    """
    返回:
        sdf_p: (B,P) or (P,) 点云处三线性插值得到的SDF
        valid_mask: (B,P) or (P,) 点是否在盒内（越界点不参与插值）
    说明:
        - 自动将 pcls 从世界坐标映射到 [0,res] 的连续体素坐标
        - 顶点格大小 V = res + 1
        - 索引自动 clamp，越界点会被标注 invalid（不参与loss）
    """
    # 统一 batch 维
    if pcls.dim() == 2:
        pcls = pcls.unsqueeze(0)  # (1,P,3)
        squeezed = True
    else:
        squeezed = False

    B, P, _ = pcls.shape
    device = pcls.device
    V = res + 1

    # sdf_d reshape 成 (V,V,V)
    if sdf_d.dim() == 1:
        sdf_d = sdf_d.view(V, V, V)
    assert sdf_d.shape == (V, V, V), f"sdf_d must be (V,V,V), got {tuple(sdf_d.shape)}"

    # 映射世界坐标 -> 连续体素坐标 g \in [0,res]
    bbox_min = torch.as_tensor(bbox_min, device=device, dtype=pcls.dtype)
    bbox_max = torch.as_tensor(bbox_max, device=device, dtype=pcls.dtype)
    scale = (bbox_max - bbox_min).clamp_min(eps)
    g = (pcls - bbox_min) / scale * res  # (B,P,3) 将partial点云约束到[0,256]

    # 有效性：在 [0,res] 内
    valid = (g >= 0 - eps).all(dim=-1) & (g <= res + eps).all(dim=-1)
    

    # 连续坐标 -> i0/i1 & t
    i0 = torch.floor(g).to(torch.long)                # (B,P,3), torch.floor(g) 取下整（向下取整）, 索引安全在 [0, res-1]
    i1 = (i0 + 1).clamp(0, res)                       # (B,P,3) [1, res]
    t  = (g - i0.to(g.dtype)).clamp(0.0, 1.0 - 1e-7)  # (B,P,3) 防止精度卡在1.0

    x0, y0, z0 = i0[..., 0], i0[..., 1], i0[..., 2] #这里应该是找到每个partial点云附近8个顶点的最小那个顶点坐标id x, y, z
    x1, y1, z1 = i1[..., 0], i1[..., 1], i1[..., 2] #这里应该是找到每个partial点云附近8个顶点的最大那个顶点坐标id x, y, z
    tx, ty, tz = t[..., 0], t[..., 1], t[..., 2] #找到partial点云（已经变成0-256范围）到小的顶点id xyz的距离； 
                                                 #后续要得到partial点云到大的顶点id xyz距离，只需要 1-tx， 1-ty， 1-tz

    # 8个角顶点的SDF（在顶点格 V=257 上取值）
    # 小心维度: sdf_d[x,y,z] 这里默认布局 (V,V,V)
    c000 = sdf_d[x0, y0, z0]
    c100 = sdf_d[x1, y0, z0]
    c010 = sdf_d[x0, y1, z0]
    c110 = sdf_d[x1, y1, z0]
    c001 = sdf_d[x0, y0, z1]
    c101 = sdf_d[x1, y0, z1]
    c011 = sdf_d[x0, y1, z1]
    c111 = sdf_d[x1, y1, z1]

    # 三线性权重
    wx0, wy0, wz0 = (1 - tx), (1 - ty), (1 - tz)
    wx1, wy1, wz1 = tx, ty, tz

    w000 = wx0 * wy0 * wz0
    w100 = wx1 * wy0 * wz0
    w010 = wx0 * wy1 * wz0
    w110 = wx1 * wy1 * wz0
    w001 = wx0 * wy0 * wz1
    w101 = wx1 * wy0 * wz1
    w011 = wx0 * wy1 * wz1
    w111 = wx1 * wy1 * wz1

    # 插值
    # print(w000)
    # print(c000)
    sdf_p = (w000 * c000 + w100 * c100 + w010 * c010 + w110 * c110 +
             w001 * c001 + w101 * c101 + w011 * c011 + w111 * c111)  # (B,P)

    # 对越界点（无效）可置为0并用 mask 过滤loss
    sdf_p = torch.where(valid, sdf_p, torch.zeros_like(sdf_p))
    # print(sdf_p)
    # print(sdf_p.shape)
    # print(aaa)

    if squeezed:
        return sdf_p[0], valid[0]
    else:
        return sdf_p, valid


def sdf_zero_level_loss(sdf_p, valid_mask, delta=0.01):
    """
    仅对有效点做 |sdf| -> 0 的 Huber/SmoothL1.
    """
    if valid_mask is None:
        return F.smooth_l1_loss(sdf_p, torch.zeros_like(sdf_p), beta=delta)
    if valid_mask.sum() == 0:
        return sdf_p.sum() * 0.0
    return F.smooth_l1_loss(
        sdf_p[valid_mask], torch.zeros_like(sdf_p[valid_mask]), beta=delta
    )

def sdf_zero_level_l2(sdf_p, valid_mask=None):
    """
    对有效位置最小化 (sdf_p)^2 的均值。
    sdf_p: (...,) 任意形状，通常 (B,P) 或 (P,)
    valid_mask: 同形状的 bool 掩码；None 表示全部有效
    """
    if valid_mask is None:
        return (sdf_p ** 2).mean()

    # 若没有有效点，返回一个与 sdf_p 连接的 0，避免断图
    if torch.count_nonzero(valid_mask) == 0:
        return sdf_p.sum() * 0.0

    # return (sdf_p[valid_mask] ** 2).mean()
    # return (torch.abs(sdf_p[valid_mask])).mean()
    return (torch.abs(sdf_p)).mean()


def sample_sdf_with_idx_grid(
    partial_pcl,                # (B,P,3) 或 (P,3) 世界坐标
    bbox_min, 
    bbox_max,         # AABB
    res: int,                   # cell 数（如 256），顶点数 V=res+1
    idx_grid_3d,                # (V,V,V) int32/int64，每个格点里是顶点 id（-1 表示无）
    tsdf_v,                     # (Vuniq,3) 形变后顶点坐标（用来取出 8 顶点坐标，若只需 SDF也可不取）
    tsdf_s,                     # (Vuniq,) 顶点 SDF（requires_grad=True 可）
    default_outside: float = -1.0,
    eps: float = 1e-6,
):
    squeezed = False
    if partial_pcl.dim() == 2:
        partial_pcl = partial_pcl.unsqueeze(0); squeezed = True
    B,P,_ = partial_pcl.shape
    dev, dt = partial_pcl.device, partial_pcl.dtype
    V = res + 1

    # 1) 世界 -> 连续体素坐标 g ∈ [0,res)
    bbox_min = torch.as_tensor(bbox_min, device=dev, dtype=dt)
    bbox_max = torch.as_tensor(bbox_max, device=dev, dtype=dt)
    scale = (bbox_max - bbox_min).clamp_min(eps)
    g = (partial_pcl - bbox_min) / scale * res
    g = g.clamp(0.0, res)

    i0 = torch.floor(g).to(torch.long)      # (B,P,3) ∈ [0,res-1]
    t  = g - i0.to(dt)                      # (B,P,3) ∈ [0,1)
    x0,y0,z0 = i0.unbind(-1)
    x1,y1,z1 = x0+1, y0+1, z0+1             # 顶点坐标 ∈ [0,res]

    # 2) 用 idx_grid_3d 取 8 角的顶点 id
    idx_grid_1d = idx_grid_3d.reshape(-1)   # (V^3,)
    def lin(xx,yy,zz): return (xx*V*V + yy*V + zz).view(-1)  # (B*P,)
    corners_lin = torch.stack([
        lin(x0,y0,z0), lin(x1,y0,z0),
        lin(x0,y1,z0), lin(x1,y1,z0),
        lin(x0,y0,z1), lin(x1,y0,z1),
        lin(x0,y1,z1), lin(x1,y1,z1),
    ], dim=-1)                               # (B*P, 8)

    vid = idx_grid_1d[corners_lin]           # (B*P,8) int
    vid = vid.view(B,P,8).to(torch.long) 

    valid_corner = (vid >= 0)
    # 3) 取 8 顶点的 SDF（以及可选坐标）
    # 为了安全索引，把无效角临时映到 0，再用 where 替换值
    # vid_safe = vid.clamp_min(0)
    vid_safe = vid
    s8 = tsdf_s[vid_safe]                    # (B,P,8)
    # print(s8[:,:20,:])
    # s8 = torch.where(valid_corner, s8, torch.full_like(s8, default_outside))

    # 可选：如果你还需要 8 个形变后坐标（比如可视化/调试）
    v8 = tsdf_v[vid_safe]                    # (B,P,8,3)
    # pcl = partial_pcl.unsqueeze(2)
    # d = torch.norm(pcl - v8, dim=-1)    # (B,P,8)
    # w = d/torch.sum(d,dim=2,keepdim=True)

    # 4) 规则网格三线性权重（基于 t）
    tx,ty,tz = t.unbind(-1)                  # (B,P)
    wx0,wy0,wz0 = (1-tx),(1-ty),(1-tz)
    wx1,wy1,wz1 = tx,ty,tz
    w = torch.stack([
        wx0*wy0*wz0, wx1*wy0*wz0,
        wx0*wy1*wz0, wx1*wy1*wz0,
        wx0*wy0*wz1, wx1*wy0*wz1,
        wx0*wy1*wz1, wx1*wy1*wz1,
    ], dim=-1)                                # (B,P,8)
    
  
    sdf_p = (w * s8).sum(dim=-1)              # (B,P)

    # 只在 8 角至少有一个有效时才算有效（也可要求“全部 8 个有效”）
    valid = valid_corner.any(dim=-1)
    # print(aaa)
    if squeezed:
        return sdf_p[0], valid[0], v8[0], vid[0]   # 也把 8 顶点坐标 / id 返回，方便你后续用
    return sdf_p, valid, v8, vid



def _kernel_weights(dist, h, kind='gauss', sigma_vox=1.5, rmax_vox=3.0):
    """
    dist: (B,P,K) 点到顶点的欧氏距离（世界坐标）
    h:    标量，细格步长（世界尺度）
    kind: 'gauss' | 'wendland' | 'tent'
    返回 w: (B,P,K)，未归一化权（已按 rmax 截断）
    """
    r = dist / (rmax_vox * h)  # 归一化半径
    within = (r <= 1.0).to(dist.dtype)

    if kind == 'gauss':
        sigma = sigma_vox * h
        w = torch.exp(-0.5 * (dist / sigma)**2)
        w = w * within
    elif kind == 'wendland':
        z = (1.0 - torch.clamp(r, 0.0, 1.0))
        w = (z**4) * (1 + 4*r) * within   # Wendland C2
    else:  # 'tent' 线性核
        w = torch.clamp(1.0 - r, min=0.0)

    return w

def sample_sdf_with_idx_grid_coarse_kernel(
    partial_pcl,                # (B,P,3) 或 (P,3) 世界坐标
    bbox_min, bbox_max,         # AABB 世界坐标
    res_fine: int,              # 细格 cell 数（如 256）；顶点数 V = res_fine + 1
    res_coarse: int = 64,       # 粗格 cell 数（如 64）
    idx_grid_3d=None,           # (V,V,V) int32/64，顶点 id（-1 表示无）；若 None 视作连续映射
    tsdf_v=None,                # (Vuniq,3) 顶点世界坐标（用于距离核）；若 None 将用规则网格坐标
    tsdf_s=None,                # (Vuniq,) 顶点 SDF（requires_grad=True 可）
    kernel: str = 'gauss',      # 'gauss' | 'wendland' | 'tent'
    sigma_vox: float = 1.5,     # 高斯 σ（单位=细格体素）
    rmax_vox: float = 8.0,      # 截断半径（单位=细格体素），覆盖 5×5×5 刚好够
    default_outside: float = 0.0, # 没有任何有效顶点时的回退值（通常设 0 或外侧常数）
    return_debug: bool = False,
    eps: float = 1e-9,
):
    """
    思路：
    - 把每个点定位到它的粗格 cell（res_coarse），找到对应的细格“块”的最小顶点索引 base = gc*step，其中 step = res_fine/res_coarse（典型=4）
    - 该块的顶点索引范围为 base..base+4（每轴），组合出 5×5×5=125 个细顶点
    - 取这些顶点的 SDF（和坐标，若 tsdf_v 提供的是形变后顶点坐标则用之），用距离核加权求和 → 点的 SDF
    - 权重做归一化；无有效顶点时返回 default_outside
    """
    squeezed = False
    if partial_pcl.dim() == 2:
        partial_pcl = partial_pcl.unsqueeze(0); squeezed = True
    B,P,_ = partial_pcl.shape
    dev, dt = partial_pcl.device, partial_pcl.dtype

    V = res_fine + 1
    step = res_fine // res_coarse   # 通常为 4
    assert res_fine % res_coarse == 0, "res_fine 必须能整除 res_coarse（例如 256 vs 64）"
    assert tsdf_s is not None, "需要 tsdf_s 顶点 SDF"

    # 世界 → 连续细格坐标 g_fine ∈ [0,res_fine]
    bbox_min = torch.as_tensor(bbox_min, device=dev, dtype=dt)
    bbox_max = torch.as_tensor(bbox_max, device=dev, dtype=dt)
    scale = (bbox_max - bbox_min).clamp_min(1e-6)
    g_f = (partial_pcl - bbox_min) / scale * res_fine
    g_f = g_f.clamp(0.0, float(res_fine))

    # 找粗格 cell 索引（0..res_coarse-1）
    g_c = g_f / step
    ic0 = torch.floor(g_c).to(torch.long).clamp(0, res_coarse-1)  # (B,P,3)

    # 该粗格对应细格块的最小顶点索引（0..res_fine-4）
    base = (ic0 * step).clamp(0, res_fine-4)  # (B,P,3)

    # 生成 0..4 的偏移并广播，得到 5×5×5 个细顶点索引（沿三轴）
    off = torch.arange(5, device=dev, dtype=torch.long)
    ix = (base[...,0,None,None,None] + off[None,None,:,None,None]).clamp(0, res_fine)
    iy = (base[...,1,None,None,None] + off[None,None,None,:,None]).clamp(0, res_fine)
    iz = (base[...,2,None,None,None] + off[None,None,None,None,:]).clamp(0, res_fine)

    # 线性索引（V,V,V） → 1D
    lin = (ix * V * V + iy * V + iz).reshape(B, P, -1)  # (B,P,125)

    # 顶点 id 映射
    if idx_grid_3d is None:
        # 若没有 idx_grid_3d，假设顶点 id 与线性索引一致（规则网格、无压缩）
        vid = lin
        valid_corner = torch.ones_like(vid, dtype=torch.bool)
    else:
        idx_grid_1d = idx_grid_3d.reshape(-1)
        vid = idx_grid_1d[lin]          # (B,P,125)
        valid_corner = (vid >= 0)

    # 取 SDF
    vid_safe = vid.clamp_min(0)
    sK = tsdf_s[vid_safe]               # (B,P,125)
    sK = torch.where(valid_corner, sK, torch.full_like(sK, default_outside))

    # 顶点坐标（用于距离核）。如果没提供 tsdf_v，则用规则网格坐标（世界尺度）
    if tsdf_v is not None:
        vK = tsdf_v[vid_safe]           # (B,P,125,3)
        # 对无效顶点可以把坐标填回点自身，避免数值炸（反正权重会是 0）
        vK = torch.where(valid_corner[...,None], vK, partial_pcl[...,None,:])
    else:
        # 构造规则网格的世界坐标
        xs = torch.linspace(bbox_min[0], bbox_max[0], V, device=dev, dtype=dt)
        ys = torch.linspace(bbox_min[1], bbox_max[1], V, device=dev, dtype=dt)
        zs = torch.linspace(bbox_min[2], bbox_max[2], V, device=dev, dtype=dt)
        # 注意：这里我们手上是每点各自的 (ix,iy,iz) 组合，直接 gather 更省
        vx = xs[ix].reshape(B,P,-1); vy = ys[iy].reshape(B,P,-1); vz = zs[iz].reshape(B,P,-1)
        vK = torch.stack([vx,vy,vz], dim=-1)  # (B,P,125,3)

    # 细格体素边长（世界尺度）
    h = (bbox_max - bbox_min) / res_fine
    # 这里三轴可能不同，取均值或 max 都可；一般是各向同性盒子，取均值
    h = h.mean().item()

    # 距离与核权
    d = torch.linalg.norm(vK - partial_pcl[...,None,:], dim=-1)  # (B,P,125)
    w = _kernel_weights(d, h, kind=kernel, sigma_vox=sigma_vox, rmax_vox=rmax_vox)
    # print(w[0,0,...])

    # 把无效顶点的权清零
    w = torch.where(valid_corner, w, torch.zeros_like(w))

    # 归一化
    w_sum = w.sum(dim=-1, keepdim=True)  # (B,P,1)
    # 如果没有任何有效邻域点，回退到 default_outside
    no_valid = (w_sum <= eps)
    w = torch.where(no_valid, torch.zeros_like(w), w / (w_sum + eps))

    # 加权得到点的 SDF
    # print(w[0,0,...])
    # print(sK[0,0,...])
    # print(aaa)
    sdf_p = (w * sK).sum(dim=-1)  # (B,P)
    sdf_p = torch.where(no_valid.squeeze(-1), torch.full_like(sdf_p, default_outside), sdf_p)

    if squeezed:
        sdf_p = sdf_p[0]

    if return_debug:
        # 方便你调权重/可视化
        out = {
            "vid": vid if not squeezed else vid[0],
            "weights": w if not squeezed else w[0],
            "vpos": vK if not squeezed else vK[0],
            "valid": valid_corner if not squeezed else valid_corner[0],
        }
        return sdf_p, out
    return sdf_p