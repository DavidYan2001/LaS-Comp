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
from dist_chamfer_3D import chamfer_3DDist
from emd_module import emdModule
from trellis_utils import (
    center_scale_from_gt_torch,
    denormalize_with_center_scale,
    normalize_with_center_scale,
)
from trellis.pipelines import TrellisTextTo3DPipeline


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

def voxel_to_pointcloud(voxel, save_path="repaint_results_bunny/partial_voxels.ply", normalize=True):
    """
    voxel: numpy array, shape (64, 64, 64) or (1, 64, 64, 64)
    save_path: 输出的 ply 文件路径
    normalize: 是否归一化到 [-0.5, 0.5]
    """
    # 去掉 batch/channel
    if voxel.ndim == 4:
        voxel = voxel[0]  # (64, 64, 64)

    

    # 获取占据点的索引 (z, y, x)
    x, y, z = np.nonzero(voxel > 0.5)

    # 拼成 (N, 3)
    points = np.stack([x, y, z], axis=-1).astype(np.float32)

    # 归一化到 [-0.5, 0.5] 区间
    if normalize:
        points = (points+0.5) / voxel.shape[0] - 0.5

    # 创建 ply 顶点数据
    vertex = np.array(
        [(p[0], p[1], p[2]) for p in points],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )

    # 保存为 PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Saved {len(points)} points to {save_path}")

def make_octa_mask_64(device='cuda', dtype=torch.float32):
    """
    生成 [1,1,64,64,64] 的体素 mask，坐标范围为 [-0.5, 0.5]（以体素中心采样）。
    在满足 (x>0, y<0, z<0) 的区域设为 0，其余设为 1。
    """
    N = 64
    # 体素中心坐标：0.5/64, 1.5/64, ..., 63.5/64 映射到 [-0.5, 0.5]
    coords_1d = (torch.arange(N, device=device, dtype=dtype) + 0.5) / N - 0.5
    # 生成网格，注意维度顺序：D(=Z), H(=Y), W(=X)
    x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij') # [64,64,64]
    

    # 条件区域：x>0 且 y<0 且 z<0
    bad = (x > 0) & (y < 1) & (z < 0)

    mask = torch.ones((1, 1, N, N, N), device=device, dtype=dtype)
    mask[0, 0][bad] = 0.0

    return mask.squeeze(0)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("./ckpt/ckpt_text_xlarge")
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

# === 文件路径 ===
text_dict = {
    "armadillo": "An armadillo with raised arms",
    "bimba": "A bimba bust",
    "cow": "A cow",
    "horse": "A horse",
    "lucy": "The angel Lucy",
    "homer": "Homer Simpson",
    "max-planck": "The head of Max-Planck.",
    "nefertiti": "Nefertiti's bust with a crown.",
    "ogre": "An ogre",
    "stanford-bunny": "A stanford-bunny figurine with a tilted head",
    "teapot": "A teapot",
    "xyzrgb_dragon": "A xyzrgb dragon statue with limbs.",
    "01184":"An outdoor trash can with wheels",
    "06127":"A plant in a large vase",
    "06830":"A children's tricycle with adult's handle" ,
    "07306":"An office trash can",
    "05452": "An outside chair",
    "06145":"A one leg square table",
    "05117":"An old chair",
    "06188":"A motorcycle",
    "07136":"A couch",
    "09639":"An executive chair"
}




rescale_ts = [4.0] 
alpha_etas = [0.1]
optimization_steps=[1]
lrs_sample=[1e-5]
        
use_norm = True

datasets = ['redwood', 'plyobj' ]

for dataset in datasets:
    if dataset == 'plyobj':
        names = ["armadillo", "bimba", "cow", "horse", "homer", "max-planck", "nefertiti", "ogre", "stanford-bunny", "teapot", "lucy", "xyzrgb_dragon"] 

    elif dataset == 'redwood':
        names = ["05452", "06145", "05117", "09639", "06127", "06830", "07306", "06188", "01184", "07136"]
        

    CD_all=0.
    EMD_all = 0.
    seed = 1
    for name in names:
        for rescale_t in rescale_ts:
            for alpha_eta in alpha_etas:
                for optimization_step in optimization_steps:
                    for lr_sample in lrs_sample:
                        outputpath = "./results/text_cond/" + dataset  
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
                                     
                                center, scale = center_scale_from_gt_torch(torch.cat([gt,xyz], dim=0))
                            else:
                                center, scale = center_scale_from_gt_torch(gt)
                            

                            xyz = normalize_with_center_scale(xyz, center, scale)


                        print("num of input points:{}".format(xyz.shape))
                        ss, _, _ = voxelize_unit_cube(xyz, resolution=resolution, open_upper_bound=False)


                        stride = 4
                        kernel = 4
                        threshold=0.0
                        M16_filled = ss.float().squeeze(0)
                        mask_down = F.avg_pool3d(M16_filled.float().unsqueeze(0), kernel_size=kernel, stride=stride)
                        mask_down = (mask_down > threshold).float()
                        


                        outputs_ss, outputs_slat = pipeline.run_lascomp(
                            mask = mask_down.unsqueeze(0).cuda().float(),
                            ss = ss.unsqueeze(0).cuda().float(),
                            alpha_eta = alpha_eta,
                            optimization_step = optimization_step,
                            lr_sample = lr_sample,
                            rescale_t = rescale_t,
                            prompt = text_dict[name],
                            seed=seed,
                            
                        )
                        coords = outputs_ss[:, 1:].float()
                        coords = (coords+0.5)/64 - 0.5
                        if use_norm:
                            coords = denormalize_with_center_scale(coords,center.cuda(), scale.cuda())

                        coords = coords[:, [0, 1, 2]]  # 变成 (x, y, z)

                        mesh_result = outputs_slat['mesh'][0]   # MeshExtractResult 对象

                        # # 取出顶点和面
                        vertices = mesh_result.vertices.cpu().numpy()   # numpy array [N, 3]
                        faces    = mesh_result.faces.cpu().numpy()       # numpy array [M, 3]

                        if dataset == "redwood":
                                # vertices = denormalize_with_center_scale(torch.tensor(vertices).cuda(),center.cuda(), scale.cuda()).cpu().numpy()
                                vertices = np.stack([
                                    vertices[:, 0],   # -x
                                    vertices[:, 1],   # z -> y
                                    vertices[:, 2],   # y -> z
                                ], axis=1)

            
                        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                        mesh.export(outputpath + '/' + name +"_output_mesh" +".glb")

                        mesh = o3d.geometry.TriangleMesh()
                        mesh.vertices = o3d.utility.Vector3dVector(vertices)
                        mesh.triangles = o3d.utility.Vector3iVector(faces)
                        K = 16384
                        output_sampled_pts = mesh.sample_points_uniformly(20*K)   # 也可以用 sample_points_poisson_disk
                        # output_sampled_pts = mesh.sample_points_uniformly(10*K)
                        pts_mesh = np.asarray(output_sampled_pts.points)           # (20K, 3)
                        pts_all  = np.vstack([pts_mesh, xyz])              # (20K+N, 3)
                        

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
