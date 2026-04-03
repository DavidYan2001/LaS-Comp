#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export (8,16,16,16) x0 latent -> tiled cube OBJ+MTL with gaps and multi-level colors.
- Per-voxel 'usemtl' (robust across viewers: MeshLab/Blender/KeyShot)
- 5-bin percentile coloring (beige -> light pink -> deep pink)
- Only-surface export by default; optional: only 3 visible sides
"""

import os, numpy as np

# ---------- helpers ----------
def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)

def score_from_channels(x0, reduce="l2"):  # (8,16,16,16) -> (16,16,16)
    if reduce == "l2":
        return np.sqrt((x0**2).sum(axis=0))
    elif reduce == "abs_mean":
        return np.abs(x0).mean(axis=0)
    else:
        raise ValueError("reduce must be 'l2' or 'abs_mean'")

def pool_2x2x2(v16, mode="avg"):          # (16,16,16) -> (8,8,8)
    v = v16.reshape(8,2, 8,2, 8,2).transpose(0,2,4,1,3,5)  # (8,8,8,2,2,2)
    return v.mean((3,4,5)) if mode=="avg" else v.max((3,4,5))

# ---------- geometry ----------
def write_cube(fv, ff, base_idx, cx, cy, cz, a):
    """
    8 角点 + 12 三角面；返回写入后的下一个起始索引
    """
    x0,x1 = cx-a, cx+a; y0,y1 = cy-a, cy+a; z0,z1 = cz-a, cz+a
    verts = [
        (x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
        (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1),
    ]
    for v in verts:
        fv.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    # 写 6 个四边形为 12 个三角
    def quad(i0,i1,i2,i3):
        ff.write(f"f {base_idx+i0} {base_idx+i1} {base_idx+i2}\n")
        ff.write(f"f {base_idx+i0} {base_idx+i2} {base_idx+i3}\n")
    quad(0,1,2,3)  # bottom
    quad(4,5,6,7)  # top
    quad(3,2,6,7)  # front
    quad(0,1,5,4)  # back
    quad(1,2,6,5)  # right
    quad(0,3,7,4)  # left
    return base_idx + 8

def write_mtl(path_mtl, palette):
    with open(path_mtl, "w") as f:
        for name, (r,g,b) in palette.items():
            f.write(f"newmtl {name}\n")
            # 提高环境光，去高光，避免灯光影响导致“发灰/只一面有色”
            f.write(f"Ka {r:.4f} {g:.4f} {b:.4f}\n")
            f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
            f.write("Ks 0.0 0.0 0.0\n")   # 无高光
            f.write("Ns 1.0\n")
            f.write("d 1.0\nTr 0.0\n")
            f.write("illum 1\n\n")       # 无镜面模型，颜色最稳

# ---------- export ----------
def export_latent_tiles_obj(
    x0, out_dir="latent_tiles_obj",
    obj_name="latent_tiles.obj", mtl_name="latent_tiles.mtl",
    reduce="l2", pool="avg",
    # 分位阈值（从低到高）。下面这组会让红色明显减少（顶层仅~2%）
    bins=(30, 40, 50, 60, 70, 80),
    export_mode="surface",   # 'surface' | 'visible3' | 'all'
    gap=0.16, scale=1.0
):
    os.makedirs(out_dir, exist_ok=True)
    path_obj = os.path.join(out_dir, obj_name)
    path_mtl = os.path.join(out_dir, mtl_name)

    x0 = to_np(x0); assert x0.shape == (8,16,16,16)
    s16 = score_from_channels(x0, reduce=reduce)
    s8  = pool_2x2x2(s16, mode=pool)

    qs = np.percentile(s8, bins).tolist()
    # 精致淡色系（米色→沙色→浅桃→柔粉→玫瑰→深玫瑰→点缀深红）

    palette = {
        "c0": (0.900, 0.875, 0.780),  # sand
        "c1": (0.930, 0.890, 0.760),  # warm beige
        "c2": (0.950, 0.820, 0.790),  # pale peach
        "c3": (0.940, 0.600, 0.700),  # soft rose
        "c4": (0.860, 0.390, 0.500),  # deep rose
    }
    write_mtl(path_mtl, palette)

    # 分位阈值（4 个阈值 -> 5 档）
    qs = np.percentile(s8, (60, 75, 88, 96)).tolist()

    # 名称顺序需与 palette 对齐；颜色数 = 阈值数 + 1
    names = ["c0","c1","c2","c3","c4"]
    assert len(names) == len(qs) + 1

    def pick_mat(v):
        # v 落在第几个阈值右侧就选第几档
        import numpy as np
        idx = np.searchsorted(qs, v, side="right")  # 0..4
        return names[idx]

    def is_surface(i,j,k):
        return (i in (0,7)) or (j in (0,7)) or (k in (0,7))

    def is_visible3(i,j,k):  # 顶/前/右
        return (j==7) or (k==7) or (i==7)

    N=8; cell=scale/N; a = cell*(1.0-gap)*0.5

    with open(path_obj, "w") as fobj:
        fobj.write("# latent tiles pastel palette\n")
        fobj.write(f"mtllib {os.path.basename(path_mtl)}\n")
        vidx = 1
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if export_mode=="surface" and not is_surface(i,j,k): continue
                    if export_mode=="visible3" and not is_visible3(i,j,k): continue
                    mat = pick_mat(s8[i,j,k])
                    fobj.write(f"o v_{i}_{j}_{k}\nusemtl {mat}\n")
                    cx = (i+0.5)*cell - scale/2.0
                    cy = (j+0.5)*cell - scale/2.0
                    cz = (k+0.5)*cell - scale/2.0
                    # 写 8 顶点 + 12 三角
                    x0,x1 = cx-a, cx+a; y0,y1 = cy-a, cy+a; z0,z1 = cz-a, cz+a
                    for vx,vy,vz in [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
                                     (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]:
                        fobj.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
                    def quad(i0,i1,i2,i3):
                        nonlocal vidx
                        fobj.write(f"f {vidx+i0} {vidx+i1} {vidx+i2}\n")
                        fobj.write(f"f {vidx+i0} {vidx+i2} {vidx+i3}\n")
                    quad(0,1,2,3); quad(4,5,6,7)
                    quad(3,2,6,7); quad(0,1,5,4)
                    quad(1,2,6,5); quad(0,3,7,4)
                    vidx += 8

# --------------- demo ---------------
if __name__ == "__main__":
    x0 = np.random.randn(8,16,16,16).astype(np.float32)
    export_latent_tiles_obj(
        x0,
        out_dir="latent_tiles_obj",
        obj_name="latent_tiles.obj",
        mtl_name="latent_tiles.mtl",
        reduce="l2", pool="avg",
        bins=(60,75,85,92),     # 调整档位能更“显形”
        export_mode="surface",  # 或 'visible3' 做论文示意
        gap=0.16, scale=1.0
    )
