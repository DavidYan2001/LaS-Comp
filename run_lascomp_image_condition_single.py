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
from PIL import Image
from plyfile import PlyData

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines import samplers


def resolve_coord_mode(dataset: str, coord_mode: str) -> str:
    if coord_mode != "auto":
        return coord_mode
    if dataset == "redwood":
        return "negx_swap_yz"
    if dataset == "plyobj":
        return "swap_yz"
    return "none"


def resolve_transform(dataset: str, coord_mode: str, yz_flip: bool | None) -> tuple[bool, bool]:
    mode = resolve_coord_mode(dataset, coord_mode)
    if mode == "none":
        x_negate = False
        default_yz_flip = False
    elif mode == "swap_yz":
        x_negate = False
        default_yz_flip = True
    elif mode == "negx_swap_yz":
        x_negate = True
        default_yz_flip = True
    else:
        raise KeyError(mode)

    if yz_flip is None:
        return x_negate, default_yz_flip
    return x_negate, yz_flip


def load_points_from_ply(
    ply_data: PlyData,
    dataset: str,
    coord_mode: str,
    yz_flip: bool | None = None,
) -> torch.Tensor:
    x = np.asarray(ply_data.elements[0]["x"])
    y = np.asarray(ply_data.elements[0]["y"])
    z = np.asarray(ply_data.elements[0]["z"])

    x_negate, use_yz_flip = resolve_transform(dataset, coord_mode, yz_flip)
    x_out = -x if x_negate else x
    if use_yz_flip:
        points = np.stack((x_out, z, y), axis=1)
    else:
        points = np.stack((x_out, y, z), axis=1)

    return torch.tensor(points, dtype=torch.float32)


def normalize_point_cloud(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xyz_min = xyz.min(dim=0).values
    xyz_max = xyz.max(dim=0).values
    center = (xyz_min + xyz_max) / 2
    scale = (xyz_max - xyz_min).max()
    scale = torch.clamp(scale, min=1e-8)
    normalized = (xyz - center) / scale
    return normalized, center, scale


def denormalize_point_cloud(xyz: torch.Tensor, center: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return xyz * scale + center


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


def build_pipeline(ckpt_path: str) -> TrellisImageTo3DPipeline:
    pipeline = TrellisImageTo3DPipeline.from_pretrained(ckpt_path)
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
        description="Run image-conditioned completion for a single partial shape."
    )
    parser.add_argument("--partial-path", type=str, required=True, help="Path to the input partial PLY.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the conditioning image.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="custom",
        help="Dataset convention used for automatic coordinate transform. Common values: custom, plyobj, redwood.",
    )
    parser.add_argument(
        "--coord-mode",
        type=str,
        default="auto",
        choices=["auto", "none", "swap_yz", "negx_swap_yz"],
        help="Coordinate transform mode applied to the input point cloud.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results. Defaults to ./results/image_cond/single/<partial_name>__<image_name>/",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="./ckpt/ckpt_image_large",
        help="Path to the TRELLIS image checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--rescale-t", type=float, default=3.0, help="Rescale timestep used in lascomp.")
    parser.add_argument("--alpha-eta", type=float, default=0.1, help="Alpha eta used in lascomp.")
    parser.add_argument("--optimization-step", type=int, default=1, help="Optimization steps in lascomp.")
    parser.add_argument("--lr-sample", type=float, default=1e-5, help="Sampling learning rate.")
    parser.add_argument(
        "--mesh-points",
        type=int,
        default=16384,
        help="Number of points to sample from the completed mesh for saving.",
    )
    parser.add_argument(
        "--normalize-partial",
        action="store_true",
        help="Normalize the partial point cloud to the unit cube using its own bounding box before inference.",
    )
    yz_flip_group = parser.add_mutually_exclusive_group()
    yz_flip_group.add_argument(
        "--yz-flip",
        dest="yz_flip",
        action="store_true",
        help="Force Y/Z swap on the partial point cloud.",
    )
    yz_flip_group.add_argument(
        "--no-yz-flip",
        dest="yz_flip",
        action="store_false",
        help="Force no Y/Z swap on the partial point cloud.",
    )
    parser.add_argument(
        "--no-preprocess-image",
        dest="preprocess_image",
        action="store_false",
        help="Disable TRELLIS image preprocessing before encoding.",
    )
    parser.set_defaults(preprocess_image=True, yz_flip=None)
    return parser.parse_args()


def main():
    args = parse_args()

    partial_path = Path(args.partial_path)
    image_path = Path(args.image_path)
    if not partial_path.exists():
        raise FileNotFoundError(partial_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if args.output_dir is None:
        output_dir = Path("./results/image_cond/test") / f"{partial_path.stem}__{image_path.stem}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plydata = PlyData.read(str(partial_path))
    xyz = load_points_from_ply(plydata, args.dataset, args.coord_mode, yz_flip=args.yz_flip)
    xyz_original = xyz.clone()

    center = None
    scale = None
    if args.normalize_partial:
        xyz, center, scale = normalize_point_cloud(xyz)

    save_point_cloud(xyz_original.cpu().numpy(), output_dir / "input_partial.ply")

    image = Image.open(image_path).convert("RGB")
    image.save(output_dir / "input_image.png")

    pipeline = build_pipeline(args.ckpt_path)

    resolution = 64
    ss, kept_idx, _ = voxelize_unit_cube(xyz, resolution=resolution, open_upper_bound=False)
    if kept_idx.numel() == 0:
        raise ValueError("No input points fall inside the unit cube after preprocessing.")
    if kept_idx.numel() < xyz.shape[0]:
        print(
            f"Warning: only {kept_idx.numel()} / {xyz.shape[0]} points are inside [-0.5, 0.5]. "
            "Consider using --normalize-partial if this is unexpected."
        )

    mask_down = F.avg_pool3d(
        ss.float().squeeze(0).unsqueeze(0),
        kernel_size=4,
        stride=4,
    )
    mask_down = (mask_down > 0.0).float()

    outputs_ss, outputs_slat = pipeline.run_lascomp(
        mask=mask_down.unsqueeze(0).cuda().float(),
        ss=ss.unsqueeze(0).cuda().float(),
        alpha_eta=args.alpha_eta,
        optimization_step=args.optimization_step,
        lr_sample=args.lr_sample,
        rescale_t=args.rescale_t,
        image=image,
        seed=args.seed,
        preprocess_image=args.preprocess_image,
    )

    sparse_coords = outputs_ss[:, 1:].float()
    sparse_coords = (sparse_coords + 0.5) / 64 - 0.5
    if args.normalize_partial:
        sparse_coords = denormalize_point_cloud(sparse_coords, center.cuda(), scale.cuda())
    save_point_cloud(sparse_coords.cpu().numpy(), output_dir / "sparse_structure.ply")

    mesh_result = outputs_slat["mesh"][0]
    vertices = mesh_result.vertices.cpu().numpy()
    faces = mesh_result.faces.cpu().numpy()

    if args.normalize_partial:
        vertices_t = torch.tensor(vertices, dtype=torch.float32, device="cuda")
        vertices = denormalize_point_cloud(vertices_t, center.cuda(), scale.cuda()).cpu().numpy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(str(output_dir / "output_mesh.glb"))

    sampled_mesh = o3d.geometry.TriangleMesh()
    sampled_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    sampled_mesh.triangles = o3d.utility.Vector3iVector(faces)

    output_sampled_pts = sampled_mesh.sample_points_uniformly(20 * args.mesh_points)
    pts_mesh = np.asarray(output_sampled_pts.points)
    pts_all = np.vstack([pts_mesh, xyz_original.cpu().numpy()])

    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(pts_all)
    fps_pcd = pcd_all.farthest_point_down_sample(args.mesh_points)
    save_point_cloud(np.asarray(fps_pcd.points), output_dir / "output_points.ply")

    with (output_dir / "run_args.txt").open("w", encoding="utf-8") as f:
        x_negate, use_yz_flip = resolve_transform(args.dataset, args.coord_mode, args.yz_flip)
        f.write(f"partial_path={partial_path}\n")
        f.write(f"image_path={image_path}\n")
        f.write(f"dataset={args.dataset}\n")
        f.write(f"coord_mode={resolve_coord_mode(args.dataset, args.coord_mode)}\n")
        f.write(f"x_negate={x_negate}\n")
        f.write(f"yz_flip={use_yz_flip}\n")
        f.write(f"normalize_partial={args.normalize_partial}\n")
        f.write(f"preprocess_image={args.preprocess_image}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"rescale_t={args.rescale_t}\n")
        f.write(f"alpha_eta={args.alpha_eta}\n")
        f.write(f"optimization_step={args.optimization_step}\n")
        f.write(f"lr_sample={args.lr_sample}\n")

    x_negate, use_yz_flip = resolve_transform(args.dataset, args.coord_mode, args.yz_flip)
    print(f"Saved results to: {output_dir}")
    print(f"Input partial: {partial_path}")
    print(f"Input image: {image_path}")
    print(f"Coord mode: {resolve_coord_mode(args.dataset, args.coord_mode)}")
    print(f"X negate: {x_negate}")
    print(f"YZ flip: {use_yz_flip}")
    print(f"Normalize partial: {args.normalize_partial}")
    print(f"Preprocess image: {args.preprocess_image}")


if __name__ == "__main__":
    main()
