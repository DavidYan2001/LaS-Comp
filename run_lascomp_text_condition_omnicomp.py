import os
from pathlib import Path

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
from plyfile import PlyData

from dist_chamfer_3D import chamfer_3DDist
from emd_module import emdModule
from eval_pcn_benchmark.trellis_utils import (
    center_scale_from_gt_torch,
    denormalize_with_center_scale,
    normalize_with_center_scale,
)
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.pipelines import samplers


def normalize_dataset_name(dataset: str) -> str:
    if dataset.startswith("omni_"):
        return dataset[len("omni_") :]
    return dataset


def load_points_from_ply(ply_data: PlyData, dataset: str) -> torch.Tensor:
    dataset_key = normalize_dataset_name(dataset)
    x = np.asarray(ply_data.elements[0]["x"])
    y = np.asarray(ply_data.elements[0]["y"])
    z = np.asarray(ply_data.elements[0]["z"])

    if dataset_key in {"redwood_preprocessed", "ycb_preprocessed"}:
        points = np.stack((x, y, z), axis=1)
    elif dataset_key == "synthetic_preprocessed":
        points = np.stack((x, -z, y), axis=1)
    else:
        raise KeyError(dataset)

    return torch.tensor(points, dtype=torch.float32)


def voxelize_unit_cube(xyz: torch.Tensor, resolution=64, open_upper_bound=True):
    device = xyz.device
    eps = 1e-6 if open_upper_bound else 0.0

    valid = (
        (xyz[:, 0] >= -0.5 + eps)
        & (xyz[:, 0] <= 0.5 - eps)
        & (xyz[:, 1] >= -0.5 + eps)
        & (xyz[:, 1] <= 0.5 - eps)
        & (xyz[:, 2] >= -0.5 + eps)
        & (xyz[:, 2] <= 0.5 - eps)
    )
    kept_idx = valid.nonzero(as_tuple=False).squeeze(1)
    xyz_kept = xyz[kept_idx]

    coords = torch.floor((xyz_kept + 0.5) * resolution).long()
    coords = torch.clamp(coords, 0, resolution - 1)

    grid = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long, device=device)
    grid[0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return grid, kept_idx, coords


def make_octa_mask_64(device="cuda", dtype=torch.float32):
    n = 64
    coords_1d = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n - 0.5
    x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    bad = (x > 0) & (y < 1) & (z < 0)

    mask = torch.ones((1, 1, n, n, n), device=device, dtype=dtype)
    mask[0, 0][bad] = 0.0
    return mask.squeeze(0)


def save_point_cloud(points, save_path: Path) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    o3d.io.write_point_cloud(str(save_path), pcd)


pipeline = TrellisTextTo3DPipeline.from_pretrained(
    "./ckpt/ckpt_text_xlarge"
)
pipeline.sparse_structure_sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
pipeline.sparse_structure_sampler_params = {
    "denoise": {
        "steps": 100,
        "cfg_strength": 1.0,
        "cfg_interval": [0.5, 1.0],
    }
}
pipeline.slat_sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
pipeline.slat_sampler_params = {
    "steps": 25,
    "cfg_strength": 1.0,
    "cfg_interval": [0.5, 1.0],
}
pipeline.cuda()

chamfer_dist = chamfer_3DDist()
emd = torch.nn.DataParallel(emdModule().cuda()).cuda()

text_dict = {
    "Robot": "A concise white Gundam robot with wings",
    "Statue": "A buddha Statue",
    "Dinosaur": "A Tyrannosaurus rex",
    "Glass": "A pair of black-framed sunglasses",
    "Guitar": "An acoustic guitar",
    "Headphone": "A red pair of headphones",
    "Laptop": "A laptop with a pink screen and edges",
    "Shoe": "A sports shoe with a textured sole and a heel loop",
    "Toilet": "A white toilet with a lid and an attached pipe",
    "Torch": "A silver cylindrical tube or pipe",
    "08310": "A bench with a backrest and four legs",
    "01027": "A bicycle with visible wheels",
    "01833": "A small mobile snack booth with food",
    "09862": "A rocking chair",
    "07089": "A motorcycle",
    "01032": "A vase with flowers",
    "07155": "A sign with a rectangular board and four supporting legs",
    "08719": "A large cushioned sofa with several pillows on it",
    "09643": "A desk with curved surfaces and legs",
    "01382": "A trash bin with a wide open top and a flared rim",
    "003": "A cracker box",
    "006": "A mustard bottle",
    "011": "A banana",
    "021": "A bleach cleanser",
    "027": "A skillet",
    "035": "A power drill",
    "052": "A large clamp",
    "056": "A tennis ball",
    "070": "A colored wood blocks",
    "076": "A timer",
}

dataset_names = {
    "ycb_preprocessed": ["003", "006", "011", "021", "027", "035", "052", "056", "070", "076"],
    "synthetic_preprocessed": [
        "Robot",
        "Statue",
        "Dinosaur",
        "Glass",
        "Guitar",
        "Headphone",
        "Laptop",
        "Shoe",
        "Toilet",
        "Torch",
    ],
    "redwood_preprocessed": [
        "08310",
        "01027",
        "01833",
        "09862",
        "07089",
        "01032",
        "07155",
        "08719",
        "09643",
        "01382",
    ],
}

rescale_ts = [3.0]
alpha_etas = [0.1]
optimization_steps = [1]
lrs_sample = [1e-5]

mask_types = ["single_scan", "semantic_part", "random_crop"]
partial_ids = ["partial_0", "partial_1"]
seeds = [1]

use_norm = False
test_with_gt = False

datasets = ["ycb_preprocessed"]
data_root = Path("./samples/omni_comp3d")
result_root = Path("./results/text_cond/omni")

for dataset in datasets:
    dataset_key = normalize_dataset_name(dataset)
    names = dataset_names[dataset_key]
    dataset_output_dir = result_root / dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = dataset_output_dir / "metrics_log.txt"

    cd_values = []
    emd_values = []

    for seed in seeds:
        for name in names:
            for mask_type in mask_types:
                for partial_id in partial_ids:
                    for rescale_t in rescale_ts:
                        for alpha_eta in alpha_etas:
                            for optimization_step in optimization_steps:
                                for lr_sample in lrs_sample:
                                    case_output_dir = dataset_output_dir / name / mask_type / partial_id
                                    case_output_dir.mkdir(parents=True, exist_ok=True)

                                    gt_datapath = data_root / dataset / name / "gt_data" / f"{name}.ply"
                                    datapath = data_root / dataset / name / "partial_data" / mask_type / f"{partial_id}.ply"

                                    gtdata = PlyData.read(str(gt_datapath))
                                    plydata = PlyData.read(str(datapath))
                                    gt = load_points_from_ply(gtdata, dataset)
                                    xyz = load_points_from_ply(plydata, dataset)

                                    if use_norm:
                                        center, scale = center_scale_from_gt_torch(gt)
                                        xyz = normalize_with_center_scale(xyz, center, scale)

                                    save_point_cloud(xyz.cpu().numpy(), case_output_dir / "input_sample.ply")
                                    print(f"num of input points: {xyz.shape}")

                                    resolution = 64
                                    ss, _, _ = voxelize_unit_cube(xyz, resolution=resolution, open_upper_bound=False)

                                    kernel = 4
                                    stride = 4
                                    threshold = 0.0
                                    mask_down = F.avg_pool3d(
                                        ss.float().squeeze(0).unsqueeze(0),
                                        kernel_size=kernel,
                                        stride=stride,
                                    )
                                    mask_down = (mask_down > threshold).float()

                                    if test_with_gt:
                                        comp_mask = make_octa_mask_64()
                                        mask_down = F.avg_pool3d(
                                            comp_mask.float(),
                                            kernel_size=kernel,
                                            stride=stride,
                                        )
                                        mask_down = (mask_down > threshold).float()

                                    _, outputs_slat = pipeline.run_lascomp(
                                        mask=mask_down.unsqueeze(0).cuda().float(),
                                        ss=ss.unsqueeze(0).cuda().float(),
                                        alpha_eta=alpha_eta,
                                        optimization_step=optimization_step,
                                        lr_sample=lr_sample,
                                        rescale_t=rescale_t,
                                        prompt=text_dict[name],
                                        seed=seed,
                                    )

                                    mesh_result = outputs_slat["mesh"][0]
                                    vertices = mesh_result.vertices.cpu().numpy()
                                    faces = mesh_result.faces.cpu().numpy()

                                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                                    mesh.export(str(case_output_dir / "output_mesh.glb"))

                                    sampled_mesh = o3d.geometry.TriangleMesh()
                                    sampled_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                                    sampled_mesh.triangles = o3d.utility.Vector3iVector(faces)

                                    num_points = 16384
                                    output_sampled_pts = sampled_mesh.sample_points_uniformly(20 * num_points)
                                    pts_mesh = np.asarray(output_sampled_pts.points)
                                    pts_all = np.vstack([pts_mesh, xyz.cpu().numpy()])

                                    pcd_all = o3d.geometry.PointCloud()
                                    pcd_all.points = o3d.utility.Vector3dVector(pts_all)
                                    fps_pcd = pcd_all.farthest_point_down_sample(num_points)
                                    fps_points = torch.tensor(
                                        np.asarray(fps_pcd.points),
                                        dtype=torch.float32,
                                        device="cuda",
                                    ).clone()

                                    if use_norm:
                                        fps_points = denormalize_with_center_scale(
                                            fps_points,
                                            center.cuda(),
                                            scale.cuda(),
                                        )

                                    d1, d2, _, _ = chamfer_dist(
                                        fps_points.unsqueeze(0),
                                        gt.cuda().unsqueeze(0),
                                    )
                                    cd = (torch.mean(torch.sqrt(d1)) + torch.mean(torch.sqrt(d2))) / 2

                                    emd_value, _ = emd(
                                        fps_points.unsqueeze(0),
                                        gt.cuda().unsqueeze(0),
                                        eps=0.005,
                                        iters=50,
                                    )
                                    emd_value = torch.sqrt(emd_value).mean(1).mean()

                                    cd_values.append(cd.item())
                                    emd_values.append(emd_value.item())

                                    result_str = (
                                        f"name={name}, mask_type={mask_type}, partial_id={partial_id}, seed={seed}, "
                                        f"CD={cd.item()}, EMD={emd_value.item()}, rescale_t={rescale_t}, "
                                        f"alpha_eta={alpha_eta}, optimization_step={optimization_step}, "
                                        f"lr_sample={lr_sample}\n"
                                    )
                                    print(result_str.strip())

                                    with metrics_log_path.open("a", encoding="utf-8") as f:
                                        f.write(result_str)

                                    save_point_cloud(
                                        fps_points.cpu().numpy(),
                                        case_output_dir / "fps_points.ply",
                                    )

    if cd_values and emd_values:
        summary_str = (
            f"dataset={dataset}, mean_CD={sum(cd_values) / len(cd_values)}, "
            f"mean_EMD={sum(emd_values) / len(emd_values)}, num_cases={len(cd_values)}\n"
        )
        print(summary_str.strip())
        with metrics_log_path.open("a", encoding="utf-8") as f:
            f.write(summary_str)
