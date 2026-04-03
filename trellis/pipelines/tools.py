import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as nd


def build_boundary_relaxed_mask(ss: torch.Tensor,
                                target_reso: int = 16,
                                sigma: float = 2.0,
                                boundary_val: float = 0.6) -> torch.Tensor:
    """
    构建 mask:
      - 内部观测点 = 1
      - 边界观测点 = Gaussian falloff (~ boundary_val~1)
      - 非观测区 = 0
    并下采样到 target_reso³

    Args:
        ss (torch.Tensor): [B,1,64,64,64] 占据体素 (0/1 float)
        target_reso (int): 输出分辨率
        sigma (float): Gaussian 衰减参数
        boundary_val (float): 边界点的最低 mask 值

    Returns:
        torch.Tensor: [B,1,target_reso,target_reso,target_reso] soft mask
    """
    B, C, D, H, W = ss.shape
    assert D == H == W == 64, "目前只支持 64³"
    assert C == 1

    device = ss.device
    hard_mask = (ss > 0).float()

    # --- Step1: 边界检测 ---
    neigh = F.max_pool3d(hard_mask, kernel_size=3, stride=1, padding=1)
    boundary = ((neigh - hard_mask) > 0).float()   # 边界观测体素
    interior = hard_mask - boundary                # 内部观测体素

    # --- Step2: 边界点 soft 处理 ---
    ss_np = hard_mask.squeeze(1).cpu().numpy().astype(np.float32)
    boundary_np = boundary.squeeze(1).cpu().numpy().astype(np.float32)

    soft_masks = []
    for b in range(B):
        occ = ss_np[b]
        dist = nd.distance_transform_edt(1 - occ)
        falloff = np.exp(-(dist**2) / (2 * sigma**2))
        # 缩放到 [boundary_val,1]
        falloff = boundary_val + (1 - boundary_val) * falloff
        # 应用规则
        soft = np.zeros_like(occ, dtype=np.float32)
        soft[occ > 0] = 1.0            # 默认观测点=1
        soft[boundary_np[b] > 0] = falloff[boundary_np[b] > 0]  # 边界点替换为 soft
        soft_masks.append(torch.from_numpy(soft))

    mask_soft = torch.stack(soft_masks, dim=0).unsqueeze(1).to(device=device, dtype=ss.dtype)

    # --- Step3: 下采样 ---
    stride = D // target_reso
    mask_down = F.avg_pool3d(mask_soft, kernel_size=stride, stride=stride)

    return mask_down.clamp(0.0, 1.0)


def build_voxel_mask_from_grid(voxel_in, k_nn=3, tau=0.3):
    """
    从 64³ voxel grid 输入构建 soft mask，并下采样到 16³
    Args:
        voxel_in: (1,1,64,64,64) tensor, 值 {0,1} 或 [0,1]
        k_nn: 邻域半径 (voxel 单位)
        tau: 密度阈值
    Returns:
        mask16: (1,1,16,16,16) tensor，值 ∈ [0,1]
    """
    # --- Step 1: 邻域密度估计 (64³) ---
    kernel = torch.ones((1, 1, 2*k_nn+1, 2*k_nn+1, 2*k_nn+1),
                        device=voxel_in.device, dtype=voxel_in.dtype)
    neigh_count = F.conv3d(voxel_in, kernel, padding=k_nn)
    neigh_norm = neigh_count / neigh_count.max().clamp(min=1)

    # --- Step 2: soft/hard 规则 (64³) ---
    soft_mask64 = torch.where(neigh_norm >= tau,
                              torch.ones_like(neigh_norm),
                              neigh_norm)

    # --- Step 3: 下采样到 16³ ---
    mask16 = F.avg_pool3d(soft_mask64, kernel_size=4, stride=4)

    return mask16


@torch.no_grad()
def build_softmask_64to16_occaware(
    voxel64: torch.Tensor,      # (1,1,64,64,64), {0,1}或[0,1]
    k_density: int = 5,         # 邻域用于密度(奇数)
    k_near: int = 5,            # 邻域用于“是否邻近观测”的门控(奇数)
    q_norm: float = 0.98,       # 归一化分位数(抗outlier)
    q_tau: float = 0.5,        # 自适应分界分位数(0.35~0.50)
    alpha: float = 1.0         # sigmoid 陡峭度
) -> torch.Tensor:
    """
    返回: (1,1,16,16,16) 的 soft mask
    - 面中间(稠密) ≈ 1
    - 边缘/空洞(稀疏) ∈ (0,1)
    - 完全空的区域 = 0
    """
    x = voxel64.clamp(0, 1)
    device, dtype = x.device, x.dtype

    # 1) 64³ 局部密度 ∈ [0,1]
    pad_d = k_density // 2
    dens64 = F.avg_pool3d(x, kernel_size=k_density, stride=1, padding=pad_d)

    # 2) 分位数归一化（抗极端）
    qv = torch.quantile(dens64.flatten(), q_norm).clamp(min=1e-6)
    dens_n = (dens64 / qv).clamp_(0, 1)

    # 3) 自适应 tau（分位数）
    tau = float(torch.quantile(dens_n.flatten(), q_tau))

    # 4) 连续映射为 soft 权重（不做硬切分）
    w64 = torch.sigmoid(alpha * (dens_n - tau))   # (1,1,64,64,64)

    # 5) 邻域占据门控：远离任何点的“纯空区域”置零
    pad_n = k_near // 2
    near_occ = F.max_pool3d(x, kernel_size=k_near, stride=1, padding=pad_n)  # {0,1}
    w64 = w64 * near_occ  # 纯空 → 0，邻近观测 → 保留 soft

    # 6) 下采样到 16³（平均保留 soft）
    mask16 = F.avg_pool3d(w64, kernel_size=4, stride=4)  # (1,1,16,16,16)

    # 7) 块级占据门控：4×4×4 内完全无点 → 该块 mask=0
    block_occ = F.max_pool3d(x, kernel_size=4, stride=4)  # {0,1} on 16³
    mask16 = mask16 * block_occ

    return mask16