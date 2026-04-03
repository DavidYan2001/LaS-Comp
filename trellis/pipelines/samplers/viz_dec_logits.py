# trellis/pipelines/samplers/viz_dec_logits.py
import numpy as np
from PIL import Image, ImageFilter

__all__ = ["render_volume_glow_pink"]

# ---------- utils ----------
def _to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _norm_percentile(v, plo=0.5, phi=99.5, eps=1e-8):
    lo = np.percentile(v, plo); hi = np.percentile(v, phi)
    return np.clip((v - lo) / max(eps, hi - lo), 0.0, 1.0)

def _colormap_pink(v01):
    """[0,1] → 粉色 RGB 渐变（深洋红→玫粉→淡粉）"""
    v = np.clip(v01, 0, 1)
    c1 = np.array([0.60, 0.00, 0.45], dtype=np.float32)
    c2 = np.array([0.95, 0.40, 0.75], dtype=np.float32)
    c3 = np.array([1.00, 0.95, 0.98], dtype=np.float32)
    mid = 0.55
    t = v
    w1 = np.clip(1.0 - np.maximum(0, (t - mid))/max(1e-6, mid), 0, 1)
    w2 = np.clip(1.0 - np.abs(t - mid)/mid,                 0, 1)
    w3 = np.clip((t - mid)/max(1e-6, 1 - mid),              0, 1)
    W  = (w1 + w2 + w3 + 1e-8)[..., None]
    return (w1[...,None]/W)*c1 + (w2[...,None]/W)*c2 + (w3[...,None]/W)*c3

# ---------- main ----------
def render_volume_glow_pink(
    logits,
    mask=None,
    out_path=None,
    axis=0,
    step=0.33,
    density_scale=13.0,
    alpha_pow=3.0,
    grad_boost=2.0,
    grad_eps=1e-3,
    render_scale=4,
    target_res=(256, 256),      # ★ 新增：目标输出分辨率
    resample="bilinear",
    glow=6.0,
    bg_color=(1,1,1),
    unsharp_amount=120,
    unsharp_radius=1.0,
    unsharp_threshold=2,
):
    import numpy as np
    from PIL import Image, ImageFilter

    v = _to_np(logits).astype(np.float32)
    if mask is not None:
        m = _to_np(mask).astype(np.float32)
        v *= np.clip(m, 0, 1)

    v = _norm_percentile(v, 0.5, 99.5)
    gx, gy, gz = np.gradient(v, axis=(2,1,0))
    gmag = np.sqrt(gx*gx + gy*gy + gz*gz)
    gmag /= (gmag.max() + 1e-8)

    if axis == 0:  vol, G = v, gmag
    elif axis == 1: vol, G = np.transpose(v,(1,0,2)), np.transpose(gmag,(1,0,2))
    else:           vol, G = np.transpose(v,(2,1,0)), np.transpose(gmag,(2,1,0))

    D,H,W = vol.shape
    Hh, Wh = int(H*render_scale), int(W*render_scale)
    out_rgb, out_a = np.zeros((Hh,Wh,3), np.float32), np.zeros((Hh,Wh), np.float32)

    RSL = Image.BILINEAR if resample == "bilinear" else Image.NEAREST
    z = 0.0
    while z < D-1:
        z0 = int(z); z1 = z0 + 1; t = z - z0
        s = (1-t)*vol[z0] + t*vol[z1]
        g = (1-t)*G[z0] + t*G[z1]

        a = np.power(np.clip(s,0,1), alpha_pow)
        a = np.clip(a + grad_boost * np.clip(g - grad_eps, 0, 1), 0, 1)
        a = 1.0 - np.exp(-density_scale * a / D * step)

        sl = Image.fromarray((s*255).astype(np.uint8)).resize((Wh,Hh), RSL)
        al = Image.fromarray((a*255).astype(np.uint8)).resize((Wh,Hh), RSL)
        s_up, a_up = np.array(sl)/255.0, np.array(al)/255.0

        rgb = _colormap_pink(s_up)
        out_rgb += (1.0 - out_a)[...,None] * (rgb * a_up[...,None])
        out_a += (1.0 - out_a) * a_up
        z += step

    img = np.clip(out_rgb + (1 - out_a)[...,None]*np.array(bg_color), 0, 1)
    im = Image.fromarray((img*255).astype(np.uint8))

    if glow > 0:
        im = Image.blend(im, im.filter(ImageFilter.GaussianBlur(radius=glow)), 0.25)
    if unsharp_amount > 0:
        im = im.filter(ImageFilter.UnsharpMask(radius=unsharp_radius,
                                               percent=int(unsharp_amount),
                                               threshold=int(unsharp_threshold)))

    # --- upscale/downscale 到目标分辨率 ---
    im = im.resize(target_res, Image.LANCZOS)
    if out_path: im.save(out_path)
    return im
