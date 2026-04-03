from typing import *
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPTextModel, AutoTokenizer
import open3d as o3d
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
import torch.nn.functional as F
from .tools import build_boundary_relaxed_mask, build_softmask_64to16_occaware

class TrellisTextTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        text_cond_model (str): The name of the text conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_text_cond_model(text_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisTextTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisTextTo3DPipeline, TrellisTextTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisTextTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_text_cond_model(args['text_cond_model'])

        return new_pipeline
    
    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings
        
    def get_cond(self, prompt: List[str]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (List[str]): The text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text(prompt)
        neg_cond = self.text_cond_model['null_cond']
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
    

    def repaint_sample_sparse_structure(
        self,
        mask,
        ss,
        alpha_eta,
        optimization_step,
        lr_sample,
        rescale_t,
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
        encoder = self.models['sparse_structure_encoder']
        flow_model = self.models['sparse_structure_flow_model']
        decoder = self.models['sparse_structure_decoder']
        gt = encoder(ss, sample_posterior=False)
        # coords = torch.argwhere(decoder(gt)>0)[:, [0, 2, 3, 4]].int()
        # print(coords.shape)
        # print(aaa)
        
        gt_ss_feat = encoder(gt_ss, sample_posterior=False)
        # feat_diff = torch.sum(torch.abs((gt  - gt_ss_feat) * mask))/(torch.sum(mask)*8)
        # mask = (mask > 0.5).expand(-1, gt.size(1), -1, -1, -1)

        # gt_masked = gt[mask]             # (N_valid,)
        # gt_ss_masked = gt_ss_feat[mask]  # (N_valid,)

        # # 将它们按通道对齐 reshape 成 (N_valid/8, 8)
        # # 这样每个位置的 8 维特征形成一个向量
        # num_channels = gt.size(1)
        # n = gt_masked.numel() // num_channels
        # gt_masked = gt_masked.view(n, num_channels)
        # gt_ss_masked = gt_ss_masked.view(n, num_channels)

        # # 计算每个 voxel 的 cosine similarity，然后取平均
        # voxel_cos = F.cosine_similarity(gt_masked, gt_ss_masked, dim=1, eps=1e-8)
        # cos_mean = voxel_cos.mean()

        # print('mean cosine similarity over valid voxels:', cos_mean.item())
        # print(aaa)
        # print("ss_sum = {}".format(torch.sum(ss)))
        # print("gt_ss_sum = {}".format(torch.sum(gt_ss)))
        # print("ss_diff = {}".format(torch.sum(gt_ss-ss)))
        # print("feat_diff = {}".format(feat_diff))
       
        mask_exp = mask.expand(-1, gt.size(1), -1, -1, -1)
        # print("gt_feat = {}".format(gt_ss_feat[mask_exp.bool()]))

        # print(gt_ss_feat[mask_exp.bool()].shape)
        # print("partial_feat = {}".format(gt[mask_exp.bool()]))

        # print(torch.mean(gt_ss_feat[mask_exp.bool()]))
        # print(torch.mean(gt[mask_exp.bool()]))
        # gt[mask_exp.bool()] = gt_ss_feat[mask_exp.bool()]
        # print(aaa)
        
        
        # sampler_params_inverse = {**self.sparse_structure_sampler_params["inverse"], **sampler_params}
        # inverse_latent = self.sparse_structure_sampler.sample_inverse(
        #     model=flow_model,
        #     gt = gt,
        #     **cond_whole,
        #     **sampler_params_inverse ,
        #     verbose=True
        # ).pred_x_t



        stride = 4
        kernel = 4
       
        
        threshold=0.0
        # mask_down = F.avg_pool3d(ss.float(), kernel_size=kernel, stride=stride)
        # mask_down = (mask_down > threshold).float()
        mask_down = mask


        print("sum of down-sampled mask:{}".format(torch.sum(mask_down)))

        

        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params_denoise = {**self.sparse_structure_sampler_params["denoise"], **sampler_params}
        sampler_params_denoise['rescale_t'] = rescale_t
        # z_s = self.sparse_structure_sampler.sample_repaint_v4(
        #     model=flow_model,
        #     noise=noise,
        #     mask = mask_down,
        #     gt = gt,
        #     inverse_latent = inverse_latent,
        #     lock_obs = True,
        #     **cond_whole,
        #     **sampler_params_denoise,
        #     verbose=True
        # ).samples
        

        # z_s = self.sparse_structure_sampler.sample_coarse_DDNM(flow_model=flow_model,
        # decoder = decoder,
        # noise=noise,
        # mask = mask_down,
        # gt_latent = gt,
        # gt_voxel = ss,
        # **cond_whole,
        # **sampler_params_denoise,
        # verbose=True).samples
        """coarse-to-fine"""
        z_s = self.sparse_structure_sampler.sample_lascomp(
            flow_model=flow_model,
            decoder=decoder,
            encoder=encoder,
            noise=noise,
            mask=mask,
            partial_voxel=ss,
            alpha_eta=alpha_eta,
            optimization_step=optimization_step,
            lr_sample=lr_sample,
            **cond,
            **sampler_params_denoise,
            verbose=True,
        ).samples


        coords_coarse = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
        print("num of predicted voxels: {}".format(coords_coarse.shape))
        coords_coarse = coords_coarse[:, 1:]
        res=64
        coarse_vox = torch.zeros((1, 1, res, res, res), dtype=torch.float32, device=coords_coarse.device)
        coarse_vox[0, 0, coords_coarse[:, 0], coords_coarse[:, 1], coords_coarse[:, 2]] = 1.0
        # z_s = gt
        # coarse_vox = ss
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
            # print(ret['mesh'][0])
            # print(ret['mesh'].shape)
            # print(aaa)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
            
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
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

    def comp_sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        partial_pcl: torch.Tensor,
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
        mesh_decoder = self.models['slat_decoder_mesh']
        
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
      
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample_slat_mesh(
            flow_model,
            mesh_decoder,
            noise,
            partial_pcl,
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
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
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
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
        # return coords

    @torch.no_grad()
    def run_repaint(
        self,
        mask,
        ss,
        gt_ss,
        partial_pcl, 
        alpha_eta,
        optimization_step,
        lr_sample,
        rescale_t,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
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
        print("The prompt is {}".format(prompt))
        cond_1 = self.get_cond(['Part of'+prompt])
        cond_2 = self.get_cond([prompt])
        torch.manual_seed(seed)
        coords = self.repaint_sample_sparse_structure(mask=mask,ss=ss, gt_ss=gt_ss, 
                                                    alpha_eta=alpha_eta,
                                                    optimization_step=optimization_step,
                                                    lr_sample=lr_sample,
                                                    rescale_t = rescale_t,
                                                    cond_partial=cond_1, cond_whole= cond_2,
                                                    num_samples=num_samples, 
                                                    sampler_params=sparse_structure_sampler_params)
        
        # slat = self.comp_sample_slat(cond_2, coords, partial_pcl, slat_sampler_params)
        slat = self.sample_slat(cond_2, coords, slat_sampler_params)
        # print(slat.feats.shape)
        # print(aaa)
        return coords, self.decode_slat(slat, formats)

        # return coords
    
    def voxelize(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        """
        Voxelize a mesh.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to voxelize.
            sha256 (str): The SHA256 hash of the mesh.
            output_dir (str): The output directory.
        """
        vertices = np.asarray(mesh.vertices)
        aabb = np.stack([vertices.min(0), vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0]).max()
        vertices = (vertices - center) / scale
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        return torch.tensor(vertices).int().cuda()

    @torch.no_grad()
    def run_variant(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            mesh (o3d.geometry.TriangleMesh): The base mesh.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        coords = self.voxelize(mesh)
        coords = torch.cat([
            torch.arange(num_samples).repeat_interleave(coords.shape[0], 0)[:, None].int().cuda(),
            coords.repeat(num_samples, 1)
        ], 1)
        torch.manual_seed(seed)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
