from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """

        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            print('success to load rgba!!!')
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
      
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.95 * 255)
        # bbox = np.argwhere(alpha == 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords


    
    def lascomp_sample_sparse_structure(
        self,
        mask,
        ss,
        gt_ss,
        alpha_eta,
        optimization_step,
        lr_sample,
        rescale_t,
        cond_partial: dict,
        cond_whole: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        encoder = self.models['sparse_structure_encoder']
        flow_model = self.models['sparse_structure_flow_model']
        decoder = self.models['sparse_structure_decoder']
        gt = encoder(ss, sample_posterior=False)
        gt_ss_feat = encoder(gt_ss, sample_posterior=False)
        stride = 4
        kernel = 4
        threshold=0.0
        alpha = 32
        M16_filled = gt_ss.float().squeeze(0)
        mask_gt = F.avg_pool3d(M16_filled.float().unsqueeze(0), kernel_size=kernel, stride=stride)
        mask_gt = (mask_gt > threshold).float()
        stride = 4
        kernel = 4
        threshold=0.0
        mask_down = mask


        
        

        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params_denoise = {**self.sparse_structure_sampler_params["denoise"], **sampler_params}
        sampler_params_denoise['rescale_t'] = rescale_t
     
        z_s = self.sparse_structure_sampler.sample_lascomp(flow_model=flow_model,
        decoder = decoder,
        encoder = encoder,
        noise=noise,
        mask = mask_down,
        partial_voxel = ss,
        alpha_eta = alpha_eta,
        optimization_step = optimization_step,
        lr_sample = lr_sample,
        **cond_whole,
        **sampler_params_denoise,
        verbose=True).samples


        coords_coarse = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
        print("num of predicted voxels: {}".format(coords_coarse.shape))
        coords_coarse = coords_coarse[:, 1:]
        res=64
        coarse_vox = torch.zeros((1, 1, res, res, res), dtype=torch.float32, device=coords_coarse.device)
        coarse_vox[0, 0, coords_coarse[:, 0], coords_coarse[:, 1], coords_coarse[:, 2]] = 1.0
        
        threshold=0.0
        refine_mask = F.avg_pool3d(coarse_vox.float(), kernel_size=kernel, stride=stride)
        refine_mask = (refine_mask > threshold).float()
        
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        print("num of predicted voxels: {}".format(coords.shape))
       

        return coords


    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        # if 'gaussian' in formats:
        #     ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        # if 'radiance_field' in formats:
        #     ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @torch.no_grad()
    def run_lascomp(
        self,
        mask,
        ss,
        gt_ss,
        partial_pcl, 
        alpha_eta,
        optimization_step,
        lr_sample,
        rescale_t,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.lascomp_sample_sparse_structure(mask=mask,ss=ss, gt_ss=gt_ss, 
                                                    alpha_eta=alpha_eta,
                                                    optimization_step=optimization_step,
                                                    lr_sample=lr_sample,
                                                    rescale_t = rescale_t,
                                                    cond_partial=cond, cond_whole= cond,
                                                    num_samples=num_samples, 
                                                    sampler_params=sparse_structure_sampler_params)
        
        # slat = self.comp_sample_slat(cond_2, coords, partial_pcl, slat_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
       
        return coords, self.decode_slat(slat, formats)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np, torch
from skimage import measure
from scipy.ndimage import gaussian_filter
from PIL import Image
import math, os

# --------- utils ----------
def _to_np(x):
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    return np.asarray(x)

def _hsv2rgb(h, s, v):
    """h,s,v in [0,1], return float RGB [0,1]"""
    i = np.floor(h*6).astype(int)
    f = h*6 - i
    p = v*(1-s); q = v*(1-f*s); t = v*(1-(1-f)*s)
    r = np.choose(i%6, [v,q,p,p,t,v])
    g = np.choose(i%6, [t,v,v,q,p,p])
    b = np.choose(i%6, [p,p,t,v,v,q])
    return np.stack([r,g,b], axis=-1)

def _save_obj_with_uv(prefix, verts_xyz, faces, uv):
    """写 OBJ/MTL（三角面，v/vt 一一对应）"""
    obj = prefix + ".obj"; mtl = prefix + ".mtl"; tex = os.path.basename(prefix) + ".png"
    with open(mtl, "w") as f:
        f.write("newmtl mat0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 1\nmap_Kd "+tex+"\n")
    with open(obj, "w") as f:
        f.write("mtllib "+os.path.basename(mtl)+"\nusemtl mat0\n")
        for x,y,z in verts_xyz: f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u,v in uv:         f.write(f"vt {u:.6f} {v:.6f}\n")
        for a,b,c in faces+1:  f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

# --------- main ----------
def paint_latent_with_random_g3flow_texture(mask, out_prefix="viz_motivation/latent_g3flow",
                                            level=0.12, tex_res=2048, seed=42,
                                            hue_wrap=1.0, noise_strength=0.08,
                                            stripe_strength=0.15, blur_sigma=6.0):
    """
    mask: (B,1,D,H,W) 或 (D,H,W)
    生成随机的 G3Flow 风格流彩纹理，并贴到 latent 表面
    输出: out_prefix.obj / .mtl / .png
    """
    import numpy as np
    from skimage import measure
    from scipy.ndimage import gaussian_filter
    from PIL import Image
    import os, math

    rng = np.random.default_rng(seed)

    m = _to_np(mask)
    if m.ndim == 5:
        m = m[0,0]
    m = (m > 0).astype(np.float32)
    D,H,W = m.shape
    if m.sum() == 0:
        raise ValueError("mask 全 0")

    # 1) marching cubes
    verts, faces, *_ = measure.marching_cubes(m, level=level)
    verts = verts / np.array([D, H, W])
    v = verts[:, [2,1,0]]

    # 2) 柱面 UV 展开
    cx, cz = v[:,0].mean(), v[:,2].mean()
    u = (np.arctan2(v[:,0]-cx, v[:,2]-cz) / (2*np.pi)) % 1.0
    vy = (v[:,1] - v[:,1].min()) / (np.ptp(v[:,1]) + 1e-8)
    uv = np.stack([u, vy], axis=1)

    # 3) 生成随机流彩纹理
    UU, VV = np.meshgrid(np.linspace(0,1,tex_res,endpoint=False),
                         np.linspace(0,1,tex_res,endpoint=False))
    hue = (UU * hue_wrap + 0.15*VV) % 1.0
    stripes = 0.5*(1.0 + np.sin(2*np.pi*(UU*3.0 + 0.35*VV)))
    hue = (hue + stripe_strength*(stripes-0.5)) % 1.0
    noise = rng.standard_normal((tex_res, tex_res)).astype(np.float32)
    noise = gaussian_filter(noise, blur_sigma)
    noise = (noise - noise.min())/(noise.max()-noise.min()+1e-8)
    hue = (hue + noise_strength*(noise-0.5)) % 1.0
    sat = 0.85*np.ones_like(hue)
    val = 0.95*np.ones_like(hue)
    tex_rgb = _hsv2rgb(hue, sat, val)
    tex_img = (np.clip(tex_rgb,0,1)*255).astype(np.uint8)
    Image.fromarray(tex_img).save(out_prefix + ".png")

    # 4) 导出 OBJ/MTL/PNG
    _save_obj_with_uv(out_prefix, v, faces, uv)
    print(f"[ok] saved:\n  {out_prefix}.obj / .mtl / .png")

