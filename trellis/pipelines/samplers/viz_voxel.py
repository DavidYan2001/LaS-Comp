#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export 64x64x64 (0/1) voxels as a tiled GRID of cubes with nice colors and gaps.
- Output: OBJ + MTL (KeyShot/Blender/Meshlab 都能稳定显示)
- Two modes:
    mode="cubes"   -> 小立方体+缝隙（默认视觉最贴近“grid”）
    mode="faces"   -> 只铺外露面的薄瓷砖（更轻量）
- For dense shapes,建议 surface_only=True（不导出内部），体积大幅缩小
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
    return np.asarray(x)

def write_mtl(path_mtl, palette, use_textures=True):
    """
    为 KeyShot 稳定：Ka/Kd 直接写颜色；再配一个 8x8 的纯色 map_Kd 纹理（有些版本更稳）。
    """
    from PIL import Image
    mtl_dir = os.path.dirname(path_mtl)
    with open(path_mtl, "w") as f:
        for name, (r,g,b) in palette.items():
            f.write(f"newmtl {name}\n")
            f.write(f"Ka {r:.4f} {g:.4f} {b:.4f}\n")
            f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
            f.write("Ks 0 0 0\nNs 1\nillum 1\nd 1.0\nTr 0.0\n")
            if use_textures:
                tex = f"{name}.png"
                Image.new("RGB", (8,8),
                          (int(r*255), int(g*255), int(b*255))
                          ).save(os.path.join(mtl_dir, tex))
                f.write(f"map_Kd {tex}\n")
            f.write("\n")

# ---------- geometry emitters ----------
def emit_cube(f, base_idx, cx, cy, cz, a):
    x0,x1 = cx-a, cx+a; y0,y1 = cy-a, cy+a; z0,z1 = cz-a, cz+a
    vs = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
          (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    for vx,vy,vz in vs: f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
    def tri(a,b,c): f.write(f"f {base_idx+a} {base_idx+b} {base_idx+c}\n")
    tri(0,1,2); tri(0,2,3)   # -z
    tri(4,5,6); tri(4,6,7)   # +z
    tri(3,2,6); tri(3,6,7)   # +y
    tri(0,1,5); tri(0,5,4)   # -y
    tri(1,2,6); tri(1,6,5)   # +x
    tri(0,3,7); tri(0,7,4)   # -x
    return base_idx + 8

def emit_face(f, base_idx, plane, i,j,k, cell, scale, inset=0.18):
    # 体素边界
    x0 = i*cell - scale/2; x1 = (i+1)*cell - scale/2
    y0 = j*cell - scale/2; y1 = (j+1)*cell - scale/2
    z0 = k*cell - scale/2; z1 = (k+1)*cell - scale/2
    L = lambda a,b,t: a*(1-t)+b*t
    t = inset
    if plane=="+x":
        x = x1; y0i,y1i = L(y0,y1,t), L(y1,y0,t); z0i,z1i = L(z0,z1,t), L(z1,z0,t)
        vs = [(x,y0i,z0i),(x,y1i,z0i),(x,y1i,z1i),(x,y0i,z1i)]
    elif plane=="-x":
        x = x0; y0i,y1i = L(y0,y1,t), L(y1,y0,t); z0i,z1i = L(z0,z1,t), L(z1,z0,t)
        vs = [(x,y0i,z0i),(x,y0i,z1i),(x,y1i,z1i),(x,y1i,z0i)]
    elif plane=="+y":
        y = y1; x0i,x1i = L(x0,x1,t), L(x1,x0,t); z0i,z1i = L(z0,z1,t), L(z1,z0,t)
        vs = [(x0i,y,z0i),(x1i,y,z0i),(x1i,y,z1i),(x0i,y,z1i)]
    elif plane=="-y":
        y = y0; x0i,x1i = L(x0,x1,t), L(x1,x0,t); z0i,z1i = L(z0,z1,t), L(z1,z0,t)
        vs = [(x0i,y,z0i),(x0i,y,z1i),(x1i,y,z1i),(x1i,y,z0i)]
    elif plane=="+z":
        z = z1; x0i,x1i = L(x0,x1,t), L(x1,x0,t); y0i,y1i = L(y0,y1,t), L(y1,y0,t)
        vs = [(x0i,y0i,z),(x1i,y0i,z),(x1i,y1i,z),(x0i,y1i,z)]
    else:  # -z
        z = z0; x0i,x1i = L(x0,x1,t), L(x1,x0,t); y0i,y1i = L(y0,y1,t), L(y1,y0,t)
        vs = [(x0i,y0i,z),(x0i,y1i,z),(x1i,y1i,z),(x1i,y0i,z)]
    for vx,vy,vz in vs: f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
    f.write(f"f {base_idx+0} {base_idx+1} {base_idx+2}\n")
    f.write(f"f {base_idx+0} {base_idx+2} {base_idx+3}\n")
    return base_idx + 4

# ---------- exporter ----------
def export_vox64_grid_obj(
    vox, out_dir="vox64_grid",
    obj_name="vox64.obj", mtl_name="vox64.mtl",
    mode="cubes",          # 'cubes'(默认) | 'faces'
    surface_only=True,     # cubes 模式建议 True（密集时极大减小面数）
    scale=1.0, gap=0.16,   # cubes：缝隙比例（0.10~0.22 都好看）
    inset=0.18,            # faces：内缩形成缝（0.15~0.22）
    # 漂亮耐看的米色系（可自行微调）
    color_on=(0.96, 0.93, 0.86),     # 占据体素（ivory-beige）
    color_on_hi=None,                # 可选高亮色，如 (0.93,0.52,0.59)
):
    """
    vox: (64,64,64) 0/1 或 bool / torch.Tensor
    """
    os.makedirs(out_dir, exist_ok=True)
    path_obj = os.path.join(out_dir, obj_name)
    path_mtl = os.path.join(out_dir, mtl_name)

    g = to_np(vox).astype(np.uint8)
    assert g.shape == (64,64,64), f"expect (64,64,64), got {g.shape}"

    # 颜色 & MTL
    palette = {"on": color_on}
    if color_on_hi is not None: palette["on_hi"] = color_on_hi
    write_mtl(path_mtl, palette, use_textures=True)

    N = 64
    cell = scale / N
    if mode == "cubes":
        a = cell*(1.0-gap)*0.5

    nbrs = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
    def is_surface(i,j,k):
        if g[i,j,k]==0: return False
        for di,dj,dk in nbrs:
            ni,nj,nk = i+di, j+dj, k+dk
            if not (0<=ni<N and 0<=nj<N and 0<=nk<N): return True
            if g[ni,nj,nk]==0: return True
        return False

    # 可选的中心高亮（让内核区域略微偏粉，体现层次）
    if color_on_hi is not None and np.any(g):
        zz,yy,xx = np.mgrid[0:N,0:N,0:N]
        c=(N-1)/2.0
        dist = -np.sqrt((xx-c)**2+(yy-c)**2+(zz-c)**2)
        thr = np.percentile(dist[g==1], 85)
        hi = (dist>=thr) & (g==1)
    else:
        hi = np.zeros_like(g, bool)

    with open(path_obj, "w") as f:
        f.write("# 64^3 voxel grid (nice cubes with gaps)\n")
        f.write(f"mtllib {os.path.basename(path_mtl)}\n")
        vidx = 1

        if mode == "faces":
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        if not is_surface(i,j,k): continue
                        f.write(f"o v_{i}_{j}_{k}\n")
                        f.write(f"usemtl {'on_hi' if hi[i,j,k] else 'on'}\n")
                        for (di,dj,dk), tag in zip(nbrs, ["-x","+x","-y","+y","-z","+z"]):
                            ni,nj,nk = i+di, j+dj, k+dk
                            empty = (ni<0 or ni>=N or nj<0 or nj>=N or nk<0 or nk>=N
                                     or g[ni,nj,nk]==0)
                            if not empty: continue
                            plane = {"+x":"+x","-x":"-x","+y":"+y","-y":"-y","+z":"+z","-z":"-z"}[tag]
                            vidx = emit_face(f, vidx, plane, i,j,k, cell, scale, inset=inset)
        else:  # cubes
            # 只放表层小立方体（密集更省面）
            mask = np.zeros_like(g, bool)
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        if is_surface(i,j,k): mask[i,j,k]=True
            count = int(mask.sum())
            if count>120000:
                print(f"[WARN] cubes={count} -> OBJ 很大，考虑 mode='faces' 或增大 gap")

            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        if not mask[i,j,k]: continue
                        f.write(f"o v_{i}_{j}_{k}\n")
                        f.write(f"usemtl {'on_hi' if hi[i,j,k] else 'on'}\n")
                        cx = (i+0.5)*cell - scale/2.0
                        cy = (j+0.5)*cell - scale/2.0
                        cz = (k+0.5)*cell - scale/2.0
                        vidx = emit_cube(f, vidx, cx, cy, cz, a)

    print(f"[OK] wrote:\n  {path_obj}\n  {path_mtl}")
    print(f"mode={mode}, surface_only={surface_only}, gap={gap}, inset={inset}, scale={scale}")

# -------- demo --------
if __name__ == "__main__":
    # 示例体素：球+少量噪点
    N=64
    z,y,x = np.mgrid[0:N,0:N,0:N]
    c=(N-1)/2.0; r=18.0
    vox = (((x-c)**2+(y-c)**2+(z-c)**2) <= r*r).astype(np.uint8)
    rng = np.random.default_rng(1); vox[tuple(rng.integers(0,N,(3,300)))] = 1

    export_vox64_grid_obj(
        vox,
        out_dir="vox64_grid_cubes",
        mode="cubes",       # 想要“砖块网格”就用 cubes；更轻量则 faces
        surface_only=True,
        gap=0.16,           # 0.14~0.20 更有“缝隙感”
        color_on=(0.910, 0.500, 0.600),   # 占据色：温润米色
        color_on_hi=None                  # 需要内区点缀就给个淡粉 (0.93,0.52,0.59)
    )
