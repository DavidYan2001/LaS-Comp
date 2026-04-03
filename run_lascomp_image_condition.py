import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
# os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from plyfile import PlyData
import trimesh
from PIL import Image
from dist_chamfer_3D import chamfer_3DDist
from emd_module import emdModule
from trellis_utils import (
    center_scale_from_gt_torch,
    denormalize_with_center_scale,
    normalize_with_center_scale,
)
from trellis.pipelines import TrellisImageTo3DPipeline


def load_points_from_ply(ply_data: PlyData, dataset: str) -> torch.Tensor:
    x = np.asarray(ply_data.elements[0]["x"])
    y = np.asarray(ply_data.elements[0]["y"])
    z = np.asarray(ply_data.elements[0]["z"])

    if dataset == "plyobj":
        points = np.stack((x, z, y), axis=1)
    elif dataset == "redwood":
        points = np.stack((-x, z, y), axis=1)
    else:
        raise KeyError(dataset)

    return torch.tensor(points, dtype=torch.float32)

def voxelize_unit_cube(xyz: torch.Tensor, resolution=64, open_upper_bound=True):
    """
    xyz: [N,3] 点云坐标（和你现在的一样，单位立方体中心在(0,0,0)）
    resolution: 体素分辨率
    open_upper_bound: True 时对上边界采用开区间 (<= 0.5 - eps)，避免 0.5 * resolution → 64 越界
    """
    device = xyz.device
    eps = 1e-6 if open_upper_bound else 0.0

    # 1) 仅保留在 [-0.5, 0.5]（或 [-0.5, 0.5-eps]）内的点
    valid = (xyz[:, 0] >= -0.5+eps) & (xyz[:, 0] <= 0.5-eps ) & \
            (xyz[:, 1] >= -0.5+eps) & (xyz[:, 1] <= 0.5-eps ) & \
            (xyz[:, 2] >= -0.5+eps) & (xyz[:, 2] <= 0.5-eps )
    kept_idx = valid.nonzero(as_tuple=False).squeeze(1)
    xyz_kept = xyz[kept_idx]  # [M,3]

    # 2) 映射到体素索引
    #    [-0.5,0.5) → [0,1) → [0,resolution)
    #    用 floor 可避免 0.5 映到 64，此外再 clamp 一次保险
    coords = torch.floor((xyz_kept + 0.5) * resolution).long()
    coords = torch.clamp(coords, 0, resolution - 1)

    # 3) 构建占据网格
    grid = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long, device=device)
    grid[0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1



    return grid, kept_idx, coords

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

pipeline = TrellisImageTo3DPipeline.from_pretrained("./ckpt/ckpt_image_large")
from trellis.pipelines import samplers
pipeline.sparse_structure_sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
pipeline.sparse_structure_sampler_params = {
    "denoise": {
        "steps": 100,
        "cfg_strength": 0.0,
        "cfg_interval": [1.5, 1.0],
    }
}
pipeline.slat_sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
pipeline.slat_sampler_params = {
    "steps": 25,
    "cfg_strength": 0.0,
    "cfg_interval": [1.5, 1.0],
}

pipeline.cuda()
chamfer_dist = chamfer_3DDist()
EMD = torch.nn.DataParallel(emdModule().cuda()).cuda()


hyperparams = {
    "01184": {
        "rescale_ts": [3.0],
        "alpha_etas": [0.1],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "05117": {
        "rescale_ts": [1.0],
        "alpha_etas": [0.02],
        "optimization_steps": [1],
        "lrs_sample": [1e-4]
    },
    "05452": {
        "rescale_ts": [5.0],
        "alpha_etas": [0.1],
        "optimization_steps": [1],
        "lrs_sample": [1e-4]
    },
    "06127": {
        "rescale_ts": [2.0],
        "alpha_etas": [0.1],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "06145": {
        "rescale_ts": [4.0],
        "alpha_etas": [0.05],
        "optimization_steps": [1],
        "lrs_sample": [1e-4]
    },
    "06188": {
        "rescale_ts": [2.0],
        "alpha_etas": [0.1],
        "optimization_steps": [1],
        "lrs_sample": [1e-4]
    },
    "06830": {
        "rescale_ts": [5.0],
        "alpha_etas": [0.02],
        "optimization_steps": [1],
        "lrs_sample": [1e-4]
    },
    "07136": {
        "rescale_ts": [4.0],
        "alpha_etas": [0.02],
        "optimization_steps": [1],
        "lrs_sample": [1e-4]
    },
    "07306": {
        "rescale_ts": [2.0],
        "alpha_etas": [0.02],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "09639": {
        "rescale_ts": [2.0],
        "alpha_etas": [0.1],
        "optimization_steps": [1],
        "lrs_sample": [1e-4]
    },
    "armadillo": {
        "rescale_ts": [1.0],
        "alpha_etas": [0.2],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "bimba": {
        "rescale_ts": [4.0],
        "alpha_etas": [0.2],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "cow": {
        "rescale_ts": [4.0],
        "alpha_etas": [0.1],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "horse": {
        "rescale_ts": [2.0],
        "alpha_etas": [0.1],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "homer": {
        "rescale_ts": [3.0],
        "alpha_etas": [0.1],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "max-planck": {
        "rescale_ts": [1.0],
        "alpha_etas": [0.2],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "nefertiti": {
        "rescale_ts": [1.0],
        "alpha_etas": [0.1],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "ogre": {
        "rescale_ts": [3.0],
        "alpha_etas": [0.1],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "stanford-bunny": {
        "rescale_ts": [5.0],
        "alpha_etas": [0.2],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "teapot": {
        "rescale_ts": [1.0],
        "alpha_etas": [0.1],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "lucy": {
        "rescale_ts": [5.0],
        "alpha_etas": [0.1],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
    "xyzrgb_dragon": {
        "rescale_ts": [1.0],
        "alpha_etas": [0.05],
        "optimization_steps": [1],
        "lrs_sample": [1e-5]
    },
}
        
use_norm = True

datasets = ['redwood' ]
            # , 'plyobj']

for dataset in datasets:
    
    if dataset == 'plyobj':
        names = ["armadillo", "bimba", "cow", "horse", "homer", "max-planck", "nefertiti", "ogre", "stanford-bunny", "teapot", "lucy", "xyzrgb_dragon"] 
        
    elif dataset == 'redwood':
        names = ["05452", "06145", "05117", "09639", "06127", "06830", "07306", "06188", "01184", "07136"]
        
    CD_all=0.
    EMD_all = 0.
    for name in names:
        
        params = hyperparams[name]
        rescale_ts = params["rescale_ts"] 
        alpha_etas = params["alpha_etas"]
        optimization_steps = params["optimization_steps"]
        lrs_sample = params["lrs_sample"]
        seeds = [1]
        for seed in seeds:
            for rescale_t in rescale_ts:
                for alpha_eta in alpha_etas:
                    for optimization_step in optimization_steps:
                        for lr_sample in lrs_sample:
                            outputpath = "./results/image_cond/" + dataset 
                            os.makedirs(outputpath, exist_ok=True)
                            resolution=64
                            gt_datapath = './samples/ComPC_datasets/'+ dataset +'/gtdata/' + name +'.ply'
                            datapath = './samples/ComPC_datasets/'+ dataset +'/indata/' + name +'.ply'

                            plydata = PlyData.read(datapath)
                            gtdata = PlyData.read(gt_datapath)
                            gt = load_points_from_ply(gtdata, dataset)
                            xyz = load_points_from_ply(plydata, dataset)

                            if use_norm:
                                if dataset == 'redwood':
                                    center, scale = center_scale_from_gt_torch(torch.cat([gt, xyz], dim=0))
                                else:
                                    center, scale = center_scale_from_gt_torch(gt)
                                xyz = normalize_with_center_scale(xyz, center, scale)
                           
                            ss, _, _ = voxelize_unit_cube(xyz, resolution=resolution, open_upper_bound=False)

                            stride = 4
                            kernel = 4
                            threshold = 0.0
                            M16_filled = ss.float().squeeze(0)      
                            mask_down = F.avg_pool3d(M16_filled.float().unsqueeze(0), kernel_size=kernel, stride=stride)
                            mask_down = (mask_down > threshold).float()


                            image_path = "./samples/image_from_compc/" + dataset + "/" + name + "_color.png"
                            image = Image.open(image_path)
                            _, outputs_slat = pipeline.run_lascomp(
                                mask = mask_down.unsqueeze(0).cuda().float(),
                                ss = ss.unsqueeze(0).cuda().float(),
                                alpha_eta = alpha_eta,
                                optimization_step = optimization_step,
                                lr_sample = lr_sample,
                                rescale_t = rescale_t,
                                image = image,
                                seed=seed,
                            )

                            mesh_result = outputs_slat['mesh'][0]   # MeshExtractResult 对象
                            vertices = mesh_result.vertices.cpu().numpy()   # numpy array [N, 3]
                            if dataset == "redwood":
                                
                                vertices = np.stack([
                                    vertices[:, 0],   # -x
                                    vertices[:, 1],   # z -> y
                                    vertices[:, 2],   # y -> z
                                ], axis=1)

                            elif dataset == "plyobj":
                                
                                vertices = np.stack([
                                    vertices[:, 0],   # -x
                                    vertices[:, 1],   # z -> y
                                    vertices[:, 2],   # y -> z
                                ], axis=1)
                            faces    = mesh_result.faces.cpu().numpy()       # numpy array [M, 3]
                            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                            mesh.export(outputpath + '/' + name +"_output_mesh" +".glb")

                            mesh = o3d.geometry.TriangleMesh()
                            mesh.vertices = o3d.utility.Vector3dVector(vertices)
                            mesh.triangles = o3d.utility.Vector3iVector(faces)
                            K = 16384
                            output_sampled_pts = mesh.sample_points_uniformly(20*K)   # 也可以用 sample_points_poisson_disk
                            pts_mesh = np.asarray(output_sampled_pts.points)           # (20K, 3)
                            pts_all  = np.vstack([pts_mesh, xyz])
                        
                            pcd_all = o3d.geometry.PointCloud()
                            pcd_all.points = o3d.utility.Vector3dVector(pts_all)
                            pcd = pcd_all.farthest_point_down_sample(K)                # -> PointCloud



                            fps_points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device="cuda").clone()
                            if use_norm:
                                fps_points = denormalize_with_center_scale(fps_points,center.cuda(), scale.cuda())

                            d1, d2, _, _ = chamfer_dist(fps_points.unsqueeze(0), gt.cuda().unsqueeze(0))
                            d1 = torch.mean(torch.sqrt(d1))
                            d2 = torch.mean(torch.sqrt(d2))
                            CD = (d1 + d2) / 2
                            CD_all += CD/len(names)


                            d1, _ = EMD(fps_points.unsqueeze(0), gt.cuda().unsqueeze(0), eps=0.005, iters=50)
                            emd = torch.sqrt(d1).mean(1).mean()
                            EMD_all += emd/len(names)

                            result_str = "name is {}, seed is {}, the L1 Chamfer Distance is {}, the Earth Mover’s Distance is {}, rescale_t is {}, alpha_eta is {}, optimization_step is {}, lr_sample is {}\n".format(name, seed, CD, emd, rescale_t, alpha_eta, optimization_step, lr_sample)
                            print(result_str.strip())

                            with open(outputpath+"/metrics_log.txt", "a") as f:
                                f.write(result_str)



                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(fps_points.cpu().numpy())
                            o3d.io.write_point_cloud(outputpath + '/' + name +"_fps_points"+'.ply', pcd)
                            
    result_str = "dataset is {},  the avg L1 Chamfer Distance is {}, the avg Earth Mover’s Distance is {}\n".format(dataset, CD_all, EMD_all)
    print(result_str.strip())


    with open(outputpath+"/metrics_log.txt", "a") as f:
        f.write(result_str)
                      
