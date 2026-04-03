import argparse
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

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.pipelines import samplers


def normalize_dataset_name(dataset: str) -> str:
    if dataset.startswith("omni_"):
        return dataset[len("omni_") :]
    return dataset


def load_points_from_ply(
    ply_data: PlyData,
    dataset: str,
    yz_flip: bool | None = None,
) -> torch.Tensor:
    dataset_key = normalize_dataset_name(dataset)
    x = np.asarray(ply_data.elements[0]["x"])
    y = np.asarray(ply_data.elements[0]["y"])
    z = np.asarray(ply_data.elements[0]["z"])

    if yz_flip is True:
        points = np.stack((x, -z, y), axis=1)
    elif yz_flip is False:
        points = np.stack((x, y, z), axis=1)
    elif dataset_key in {"redwood_preprocessed", "ycb_preprocessed", "custom"}:
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


def save_point_cloud(points, save_path: Path) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    o3d.io.write_point_cloud(str(save_path), pcd)


def build_pipeline(ckpt_path: str) -> TrellisTextTo3DPipeline:
    pipeline = TrellisTextTo3DPipeline.from_pretrained(ckpt_path)
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
    return pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run text-conditioned completion for a single partial shape."
    )
    parser.add_argument("--partial-path", type=str, required=True, help="Path to the input partial PLY.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for completion.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="custom",
        help="Dataset convention used for axis ordering. Options include custom, ycb_preprocessed, redwood_preprocessed, synthetic_preprocessed, and omni_* variants.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save completion results. Defaults to ./results/text_cond/omni_single/<partial_name>/",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="./ckpt/ckpt_text_xlarge",
        help="Path to the TRELLIS text checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--rescale-t", type=float, default=3.0, help="Rescale timestep used in repaint.")
    parser.add_argument("--alpha-eta", type=float, default=0.1, help="Alpha eta used in repaint.")
    parser.add_argument("--optimization-step", type=int, default=1, help="Optimization steps in repaint.")
    parser.add_argument("--lr-sample", type=float, default=1e-5, help="Sampling learning rate.")
    parser.add_argument(
        "--mesh-points",
        type=int,
        default=16384,
        help="Number of points to sample from the completed mesh for saving.",
    )
    yz_flip_group = parser.add_mutually_exclusive_group()
    yz_flip_group.add_argument(
        "--yz-flip",
        dest="yz_flip",
        action="store_true",
        help="Force coordinate transform (x, y, z) -> (x, -z, y).",
    )
    yz_flip_group.add_argument(
        "--no-yz-flip",
        dest="yz_flip",
        action="store_false",
        help="Force no YZ flip and keep coordinates as (x, y, z).",
    )
    parser.set_defaults(yz_flip=None)
    return parser.parse_args()


def main():
    args = parse_args()

    partial_path = Path(args.partial_path)
    if not partial_path.exists():
        raise FileNotFoundError(partial_path)

    if args.output_dir is None:
        output_dir = Path("./results/text_cond/test") / partial_path.stem
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plydata = PlyData.read(str(partial_path))
    xyz = load_points_from_ply(plydata, args.dataset, yz_flip=args.yz_flip)
    save_point_cloud(xyz.cpu().numpy(), output_dir / "input_sample.ply")

    with (output_dir / "prompt.txt").open("w", encoding="utf-8") as f:
        f.write(args.prompt + "\n")

    pipeline = build_pipeline(args.ckpt_path)

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

    outputs_ss, outputs_slat = pipeline.run_lascomp(
        mask=mask_down.unsqueeze(0).cuda().float(),
        ss=ss.unsqueeze(0).cuda().float(),
        alpha_eta=args.alpha_eta,
        optimization_step=args.optimization_step,
        lr_sample=args.lr_sample,
        rescale_t=args.rescale_t,
        prompt=args.prompt,
        seed=args.seed,
    )

    sparse_coords = outputs_ss[:, 1:].float()
    sparse_coords = (sparse_coords + 0.5) / 64 - 0.5
    save_point_cloud(sparse_coords.cpu().numpy(), output_dir / "sparse_structure.ply")

    mesh_result = outputs_slat["mesh"][0]
    vertices = mesh_result.vertices.cpu().numpy()
    faces = mesh_result.faces.cpu().numpy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(str(output_dir / "output_mesh.glb"))

    sampled_mesh = o3d.geometry.TriangleMesh()
    sampled_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    sampled_mesh.triangles = o3d.utility.Vector3iVector(faces)

    output_sampled_pts = sampled_mesh.sample_points_uniformly(20 * args.mesh_points)
    pts_mesh = np.asarray(output_sampled_pts.points)

    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(pts_mesh)
    fps_pcd = pcd_all.farthest_point_down_sample(args.mesh_points)
    save_point_cloud(np.asarray(fps_pcd.points), output_dir / "output_points.ply")

    print(f"Saved results to: {output_dir}")
    print(f"Input partial: {partial_path}")
    print(f"Prompt: {args.prompt}")
    print(f"YZ flip: {args.yz_flip if args.yz_flip is not None else 'auto'}")


if __name__ == "__main__":
    main()
