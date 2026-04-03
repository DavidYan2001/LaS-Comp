import os, numpy as np


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


def export_vox64_grid_obj(
    vox, mask=None,                      # mask 与 vox 同形状
    out_dir="vox64_grid",
    obj_name="vox64.obj", mtl_name="vox64.mtl",
    mode="cubes",                        # 'cubes' | 'faces'
    surface_only=True,                   # cubes 模式建议 True
    scale=1.0, gap=0.16,                 # cubes：缝隙
    inset=0.18,                          # faces：内缩形成缝
    # 颜色：KeyShot 友好的色系（可调）
    color_bg=(0.965, 0.935, 0.865),      # 背景占据色：mask==0 & vox==1
    color_fg=(0.680, 0.860, 0.700),      # 前景占据色：mask==1 & vox==1（浅绿）
    flip_x=False                         # NEW: 关于 X 轴镜像导出
):
    """
    vox : (64,64,64) 0/1 或 bool 或 torch.Tensor
    mask: (64,64,64) 0/1 或 bool 或 torch.Tensor；None 表示全 0
    """
    import numpy as np, os
    os.makedirs(out_dir, exist_ok=True)
    path_obj = os.path.join(out_dir, obj_name)
    path_mtl = os.path.join(out_dir, mtl_name)

    g = to_np(vox).astype(np.uint8)
    assert g.shape == (64,64,64), f"expect (64,64,64), got {g.shape}"

    if mask is None:
        m = np.zeros_like(g, dtype=np.uint8)
    else:
        m = to_np(mask).astype(np.uint8)
        assert m.shape == g.shape, f"mask shape {m.shape} mismatch vox {g.shape}"

    # 两种占据类别：bg(普通) / fg(高亮)
    occ_bg = (g==1) & (m==0)
    occ_fg = (g==1) & (m==1)

    # 写 MTL（Ka/Kd + 实色 map_Kd，KeyShot 更稳）
    palette = {"bg": color_bg, "fg": color_fg}
    write_mtl(path_mtl, palette, use_textures=True)

    N = 64
    cell = scale / N
    if mode == "cubes":
        a = cell*(1.0-gap)*0.5

    nbrs = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
    def is_surface_bool(mask_occ, i,j,k):
        if not mask_occ[i,j,k]: return False
        for di,dj,dk in nbrs:
            ni,nj,nk = i+di, j+dj, k+dk
            if not (0<=ni<N and 0<=nj<N and 0<=nk<N): return True
            if not mask_occ[ni,nj,nk]: return True
        return False

    # --------- 本地写点/面（考虑 x 翻转 + 绕序） ---------
    def _write_v(f, x, y, z):
        if flip_x: x = -x
        f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

    def _write_tri(f, a, b, c, base_idx):
        if flip_x:
            # 镜像会颠倒法线；交换 b,c 还原面朝外
            f.write(f"f {base_idx+a} {base_idx+c} {base_idx+b}\n")
        else:
            f.write(f"f {base_idx+a} {base_idx+b} {base_idx+c}\n")

    def emit_cube_local(f, base_idx, cx, cy, cz, a):
        x0,x1 = cx-a, cx+a; y0,y1 = cy-a, cy+a; z0,z1 = cz-a, cz+a
        vs = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
              (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
        for vx,vy,vz in vs: _write_v(f, vx, vy, vz)
        def tri(i,j,k): _write_tri(f, i,j,k, base_idx)
        tri(0,1,2); tri(0,2,3)   # -z
        tri(4,5,6); tri(4,6,7)   # +z
        tri(3,2,6); tri(3,6,7)   # +y
        tri(0,1,5); tri(0,5,4)   # -y
        tri(1,2,6); tri(1,6,5)   # +x
        tri(0,3,7); tri(0,7,4)   # -x
        return base_idx + 8

    def emit_face_local(f, base_idx, plane, i,j,k, cell, scale, inset=0.18):
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
        else:  # "-z"
            z = z0; x0i,x1i = L(x0,x1,t), L(x1,x0,t); y0i,y1i = L(y0,y1,t), L(y1,y0,t)
            vs = [(x0i,y0i,z),(x0i,y1i,z),(x1i,y1i,z),(x1i,y0i,z)]
        for vx,vy,vz in vs: _write_v(f, vx, vy, vz)
        _write_tri(f, 0,1,2, base_idx)
        _write_tri(f, 0,2,3, base_idx)
        return base_idx + 4

    with open(path_obj, "w") as f:
        f.write("# 64^3 voxel grid with mask-colored cubes (flip_x supported)\n")
        f.write(f"mtllib {os.path.basename(path_mtl)}\n")
        vidx = 1

        if mode == "faces":
            # 只导出外露面（bg/fg 合并判断占据）
            occ_any = occ_bg | occ_fg
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        if not occ_any[i,j,k]: continue
                        if not is_surface_bool(occ_any, i,j,k): continue
                        mat = "fg" if occ_fg[i,j,k] else "bg"
                        f.write(f"o v_{i}_{j}_{k}\nusemtl {mat}\n")
                        # 对每个方向，如邻居为空则加一块面砖
                        for (di,dj,dk), tag in zip(nbrs, ["-x","+x","-y","+y","-z","+z"]):
                            ni,nj,nk = i+di, j+dj, k+dk
                            empty = (ni<0 or ni>=N or nj<0 or nj>=N or nk<0 or nk>=N
                                     or not occ_any[ni,nj,nk])
                            if not empty: continue
                            plane = {"+x":"+x","-x":"-x","+y":"+y","-y":"-y","+z":"+z","-z":"-z"}[tag]
                            vidx = emit_face_local(f, vidx, plane, i,j,k, cell, scale, inset=inset)

        else:  # "cubes"
            # 只放表层小立方体（bg/fg 合并判断表层）
            occ_any = occ_bg | occ_fg
            surf = np.zeros_like(g, dtype=bool)
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        if is_surface_bool(occ_any, i,j,k):
                            surf[i,j,k] = True

            count = int(surf.sum())
            if count > 120000:
                print(f"[WARN] cubes={count} -> OBJ 很大，考虑 mode='faces' 或增大 gap")

            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        if not surf[i,j,k]: continue
                        mat = "fg" if occ_fg[i,j,k] else "bg"
                        f.write(f"o v_{i}_{j}_{k}\nusemtl {mat}\n")
                        cx = (i+0.5)*cell - scale/2.0
                        cy = (j+0.5)*cell - scale/2.0
                        cz = (k+0.5)*cell - scale/2.0
                        vidx = emit_cube_local(f, vidx, cx, cy, cz, a)

    print(f"[OK] wrote:\n  {path_obj}\n  {path_mtl}")
    print(f"flip_x={flip_x}, mode={mode}, surface_only={surface_only}, gap={gap}, inset={inset}, scale={scale}")

