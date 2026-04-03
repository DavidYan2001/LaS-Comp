#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Bright & fast: 6-face colored Gaussian speckle cube, no shading, transparent BG.

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt

# --- erf 兼容 ---
try:
    from numpy import special as _npsp
    _erf = _npsp.erf
except Exception:
    from math import erf as _math_erf
    _erf = np.vectorize(_math_erf)

def gaussian_cdf01(x):  # N(0,1) -> [0,1]
    return 0.5 * (1.0 + _erf(x / sqrt(2.0)))

def make_rgb_gauss(res=256, clip=3.0, contrast=1.2, min_lum=0.55, gamma=0.85, seed=0):
    """
    生成亮一些的 RGB 高斯噪声纹理：
      - 三通道 ~ N(0,1) → CDF 映射到 [0,1]
      - 亮度提升：先做 gamma (<1 提升亮部)，再抬底 min_lum（>=0.5 更亮）
    """
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((res, res, 3)).astype(np.float32)
    g = np.clip(g, -clip, clip)
    rgb = gaussian_cdf01(g)                            # [0,1]
    if contrast != 1.0:
        rgb = 0.5 + (rgb - 0.5) * contrast
        rgb = np.clip(rgb, 0.0, 1.0)
    # 提亮：gamma + lift
    rgb = np.clip(rgb, 0.0, 1.0) ** gamma
    rgb = min_lum + (1.0 - min_lum) * rgb
    rgb = np.clip(rgb, 0.0, 1.0)
    rgba = np.concatenate([rgb, np.ones((*rgb.shape[:2],1), np.float32)], -1)
    return rgba

def render_color_noise_cube_full6_bright(
    out_path="color_noise_cube_bright.png",
    face_res=256, clip=3.0, contrast=1.2, min_lum=0.55, gamma=0.85,
    elev=24, azim=-38, figsize=(4.5,4.5), dpi=300,
    seed=7, same_texture=False
):
    # 6 张纹理
    tex = [make_rgb_gauss(face_res, clip, contrast, min_lum, gamma, seed+i) for i in range(6)]
    if same_texture:
        tex = [tex[0]]*6
    (tex_back, tex_left, tex_bottom, tex_front, tex_right, tex_top) = tex

    # 网格
    u = np.linspace(0,1,face_res); U, V = np.meshgrid(u, u)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax  = fig.add_subplot(111, projection='3d')

    # 远->近，shade=False 去掉暗面阴影；edgecolor='none' 更干净更快
    kw = dict(rstride=1, cstride=1, edgecolor='none', antialiased=False, shade=False)
    ax.plot_surface(U, V, np.zeros_like(U), facecolors=tex_back, **kw)   # back z=0
    ax.plot_surface(np.zeros_like(U), U, V, facecolors=tex_left, **kw)   # left x=0
    ax.plot_surface(U, np.zeros_like(U), V, facecolors=tex_bottom, **kw) # bottom y=0
    ax.plot_surface(U, V, np.ones_like(U), facecolors=tex_front, **kw)   # front z=1
    ax.plot_surface(np.ones_like(U), U, V, facecolors=tex_right, **kw)   # right x=1
    ax.plot_surface(U, np.ones_like(U), V, facecolors=tex_top, **kw)     # top y=1

    # 视角/外观
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1,1,1))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis): axis.set_ticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.grid(False)
    ax.set_facecolor((1,1,1,0)); fig.patch.set_alpha(0.0)

    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close()
    print(f"saved: {out_path}")

if __name__ == "__main__":
    render_color_noise_cube_full6_bright(
        "color_noise_cube_bright.png",
        face_res=256,  # 更快；想更快可用 192
        clip=3.0, contrast=1.25,
        min_lum=0.58,  # 决定底部亮度（0.55~0.65）
        gamma=0.80,    # <1 更亮
        elev=24, azim=-38, dpi=300,
        seed=42, same_texture=False
    )
