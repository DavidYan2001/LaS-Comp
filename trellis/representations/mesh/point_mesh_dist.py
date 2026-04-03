# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function
from torch.autograd.function import once_differentiable


"""
This file defines distances between meshes and pointclouds.
The functions make use of the definition of a distance between a point and
an edge segment or the distance of a point and a triangle (face).

The exact mathematical formulations and implementations of these
distances can be found in `csrc/utils/geometry_utils.cuh`.
"""

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3


# PointFaceDistance
class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


point_face_distance = _PointFaceDistance.apply


# FacePointDistance
class _FacePointDistance(Function):
    """
    Torch autograd Function wrapper FacePointDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_tris,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_tris: Scalar equal to maximum number of faces in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(T,)`, where `dists[t]` is the squared
                euclidean distance of `t`-th triangular face to the closest point in the
                corresponding example in the batch
            idxs: LongTensor of shape `(T,)` indicating the closest point in the
                corresponding example in the batch.

            `dists[t] = d(points[idxs[t]], tris[t, 0], tris[t, 1], tris[t, 2])`,
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`.
        """
        dists, idxs = _C.face_point_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.face_point_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


face_point_distance = _FacePointDistance.apply


# PointEdgeDistance
class _PointEdgeDistance(Function):
    """
    Torch autograd Function wrapper PointEdgeDistance Cuda implementation
    """

    @staticmethod
    def forward(ctx, points, points_first_idx, segms, segms_first_idx, max_points):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index for each example in the mesh
            segms: FloatTensor of shape `(S, 2, 3)` of edge segments. The `s`-th
                edge segment is spanned by `(segms[s, 0], segms[s, 1])`
            segms_first_idx: LongTensor of shape `(N,)` indicating the first edge
                index for each example in the mesh
            max_points: Scalar equal to maximum number of points in the batch
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest edge in the
                corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest edge in the
                corresponding example in the batch.

            `dists[p] = d(points[p], segms[idxs[p], 0], segms[idxs[p], 1])`,
            where `d(u, v0, v1)` is the distance of point `u` from the edge segment
            spanned by `(v0, v1)`.
        """
        dists, idxs = _C.point_edge_dist_forward(
            points, points_first_idx, segms, segms_first_idx, max_points
        )
        ctx.save_for_backward(points, segms, idxs)
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, segms, idxs = ctx.saved_tensors
        grad_points, grad_segms = _C.point_edge_dist_backward(
            points, segms, idxs, grad_dists
        )
        return grad_points, None, grad_segms, None, None


point_edge_distance = _PointEdgeDistance.apply


# EdgePointDistance
class _EdgePointDistance(Function):
    """
    Torch autograd Function wrapper EdgePointDistance Cuda implementation
    """

    @staticmethod
    def forward(ctx, points, points_first_idx, segms, segms_first_idx, max_segms):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index for each example in the mesh
            segms: FloatTensor of shape `(S, 2, 3)` of edge segments. The `s`-th
                edge segment is spanned by `(segms[s, 0], segms[s, 1])`
            segms_first_idx: LongTensor of shape `(N,)` indicating the first edge
                index for each example in the mesh
            max_segms: Scalar equal to maximum number of edges in the batch
        Returns:
            dists: FloatTensor of shape `(S,)`, where `dists[s]` is the squared
                euclidean distance of `s`-th edge to the closest point in the
                corresponding example in the batch
            idxs: LongTensor of shape `(S,)` indicating the closest point in the
                corresponding example in the batch.

            `dists[s] = d(points[idxs[s]], edges[s, 0], edges[s, 1])`,
            where `d(u, v0, v1)` is the distance of point `u` from the segment
            spanned by `(v0, v1)`.
        """
        dists, idxs = _C.edge_point_dist_forward(
            points, points_first_idx, segms, segms_first_idx, max_segms
        )
        ctx.save_for_backward(points, segms, idxs)
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, segms, idxs = ctx.saved_tensors
        grad_points, grad_segms = _C.edge_point_dist_backward(
            points, segms, idxs, grad_dists
        )
        return grad_points, None, grad_segms, None, None


edge_point_distance = _EdgePointDistance.apply



# [docs]
def point_mesh_edge_distance(meshes: Meshes, pcls: Pointclouds):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_edge(mesh, pcl) + edge_point(mesh, pcl)`

    `point_edge(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest edge segment in mesh and averages across all points in pcl
    `edge_point(mesh, pcl)`: Computes the squared distance of each edge segment in mesh
        to the closest point in pcl and averages across all edges in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        loss: The `point_edge(mesh, pcl) + edge_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for edges
    verts_packed = meshes.verts_packed()
    edges_packed = meshes.edges_packed()
    segms = verts_packed[edges_packed]  # (S, 2, 3)
    segms_first_idx = meshes.mesh_to_edges_packed_first_idx()
    max_segms = meshes.num_edges_per_mesh().max().item()

    # point to edge distance: shape (P,)
    point_to_edge = point_edge_distance(
        points, points_first_idx, segms, segms_first_idx, max_points
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i), )
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    weights_p = 1.0 / weights_p.float()
    point_to_edge = point_to_edge * weights_p
    point_dist = point_to_edge.sum() / N

    # edge to edge distance: shape (S,)
    edge_to_point = edge_point_distance(
        points, points_first_idx, segms, segms_first_idx, max_segms
    )

    # weight each example by the inverse of number of edges in the example
    segm_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(S_n),)
    num_segms_per_mesh = meshes.num_edges_per_mesh()  # (N,)
    weights_s = num_segms_per_mesh.gather(0, segm_to_mesh_idx)
    weights_s = 1.0 / weights_s.float()
    edge_to_point = edge_to_point * weights_s
    edge_dist = edge_to_point.sum() / N

    return point_dist + edge_dist





# [docs]
def point_mesh_face_distance(
    meshes: Meshes,
    pcls: Pointclouds,
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )
    print(point_to_face)
    print(torch.max(point_to_face))
    print(aaa)

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    weights_p = 1.0 / weights_p.float()
    point_to_face = point_to_face * weights_p
    point_dist = point_to_face.sum() / N

    # face to point distance: shape (T,)
    face_to_point = face_point_distance(
        points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
    )

    # weight each example by the inverse of number of faces in the example
    tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()  # (sum(T_n),)
    num_tris_per_mesh = meshes.num_faces_per_mesh()  # (N, )
    weights_t = num_tris_per_mesh.gather(0, tri_to_mesh_idx)
    weights_t = 1.0 / weights_t.float()
    face_to_point = face_to_point * weights_t
    face_dist = face_to_point.sum() / N

    return point_dist + face_dist



import torch
# def point_to_face_only(meshes: Meshes,
#                        pcls: Pointclouds,
#                        min_triangle_area: float = 1e-9) -> torch.Tensor:
#     """
#     单向：观测点云 → 预测网格面 的平均平方距离（按每个样本的点数做反比加权）。
#     """
#     if len(meshes) != len(pcls):
#         raise ValueError("meshes and pointclouds must be equal sized batches")
#     N = len(meshes)

#     # 点云打包
#     points = pcls.points_packed()                           # (P, 3)
#     points_first_idx = pcls.cloud_to_packed_first_idx()     # (N,)
#     max_points = pcls.num_points_per_cloud().max().item()

#     # 面（三角形）打包
#     verts_packed = meshes.verts_packed()                    # (sum Nv, 3)
#     faces_packed = meshes.faces_packed()                    # (sum Nf, 3)
#     tris = verts_packed[faces_packed]                       # (T, 3, 3)
#     tris_first_idx = meshes.mesh_to_faces_packed_first_idx()# (N,)
#     max_tris = meshes.num_faces_per_mesh().max().item()     # 未直接用，但与内部一致

#     # 点→面距离 (P,)
#     p2f = point_face_distance(points, points_first_idx,
#                               tris, tris_first_idx,
#                               max_points, min_triangle_area)

#     # 按每个样本点数做反比加权，再对 batch 求均值
#     point_to_cloud_idx = pcls.packed_to_cloud_idx()         # (P,)
#     num_points_per_cloud = pcls.num_points_per_cloud().float()  # (N,)
#     weights_p = 1.0 / num_points_per_cloud.gather(0, point_to_cloud_idx)  # (P,)
#     print(weights_p)
#     print(p2f)
#     p2f = p2f * weights_p
#     loss = p2f.sum() / max(N, 1)
#     # print(N)
#     return loss


def point_to_face_only(
    meshes: Meshes,
    pcls: Pointclouds,
    min_triangle_area: float = 1e-9,
):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    V = meshes.verts_packed()
    F = meshes.faces_packed()
    A_med, _ = median_face_area(V, F)
    tau_abs = 1e-10
    # 阈值退火：从1e-4*中位数 -> 2e-3*中位数
    c0, c1 = 1e-4, 2e-3
    prog_t = 0.5
    c_t = (1 - prog_t) * c0 + prog_t * c1
    min_area = max(tau_abs, (c_t * A_med).item())
    print("min_area=={}".format(min_area))

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )
    point_to_face = torch.sqrt(point_to_face)
    # print(torch.max(point_to_face))
    # print(torch.min(point_to_face))
    # print(aa)
   
    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    weights_p = 1.0 / weights_p.float()
    point_to_face = point_to_face * weights_p
    point_dist = point_to_face.sum() / N

    

    return point_dist


def median_face_area(verts, faces):
    tri = verts[faces.long()]                              # [T,3,3]
    e1  = tri[:,1]-tri[:,0]
    e2  = tri[:,2]-tri[:,0]
    area = 0.5 * (e1.cross(e2, dim=-1)).norm(dim=-1)      # [T]
    # 防NaN
    area = torch.clamp(area, min=0)
    med  = torch.median(area)
    return med, area


def mean_nn_distance(pcl: torch.Tensor, verts: torch.Tensor):
    """
    pcl:   [N, 3]
    verts: [V, 3]
    return: mean distance (scalar), indices [N] of nearest verts
    """
    # [N, V]
    dists = torch.cdist(pcl, verts, p=2)
    # 每个点的最近顶点距离与索引
    min_dists, nn_idx = dists.min(dim=1)           # [N], [N]
    return min_dists.mean(), nn_idx