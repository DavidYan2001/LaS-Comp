import torch
from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import *
from .flexicubes.flexicubes import FlexiCubes


class MeshExtractResult:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None,
        res=64
    ):
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.res = res
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None
        
    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        # print(face_normals.min(), face_normals.max(), face_normals.shape)
        return face_normals[:, None, :].repeat(1, 3, 1)
                
    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals   


class SparseFeatures2Mesh:
    def __init__(self, device="cuda", res=64, use_color=True):
        '''
        a model to generate a mesh from sparse features structures using flexicube
        '''
        super().__init__()
        self.device=device
        self.res = res
        self.mesh_extractor = FlexiCubes(device=device)
        self.sdf_bias = -1.0 / res
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)
        self.reg_v = verts.to(self.device)
        self.use_color = use_color
        self._calc_layout()
    
    def _calc_layout(self):
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21}
        }
        if self.use_color:
            '''
            6 channel color including normal map
            '''
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start
        
    def get_layout(self, feats : torch.Tensor, name : str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name]['shape'])
    
    def __call__(self, cubefeats : SparseTensor, training=False):
        """
        Generates a mesh based on the specified sparse voxel structures.
        Args:
            cube_attrs [Nx21] : Sparse Tensor attrs about cube weights
            verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal 
        Returns:
            return the success tag and ni you loss, 
        """
        # add sdf bias to verts_attrs
        coords = cubefeats.coords[:, 1:]
        # pts = voxel_idx_to_points(coords.cpu().numpy(), 256, bbox_min=(-0.5,-0.5,-0.5), bbox_max=(0.5,0.5,0.5), order='xyz')
        # save_ply_xyz("sparse_voxels_centers.ply", pts)

        feats = cubefeats.feats
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
     
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        # print(v_attrs.shape)
        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True)
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None

        # mask = (sdf_d != 1)   # bool tensor
        # indices = mask.nonzero(as_tuple=False)   # [K, 3]，每一行是 (z,y,x) 下标
        # values = sdf_d[mask]  # [K]，对应的 SDF 值
        # print("非 1 的元素数量:", values.numel())
        # print(values)
        # print(torch.abs(values).mean())

        # import numpy as np, mcubes
        # verts, faces = mcubes.marching_cubes(sdf_d.view(self.res+1,self.res+1,self.res+1).detach().float().cpu().numpy(), 0.0)   # verts 在体素索引坐标系

        # # 映射到世界坐标（按你的体素覆盖范围改这里）
        # R = sdf.shape[0] - 1
        # bbox_min = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        # bbox_max = np.array([ 0.5,  0.5,  0.5], dtype=np.float32)
        # verts_world = bbox_min + (verts / R) * (bbox_max - bbox_min)
        # save_ply("sdf_iso0.ply", verts_world.astype(np.float32), faces.astype(np.int32))

        # print(aaa)

     
        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)
        
        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=self.reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            training=training)
        
        mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
     
        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:,:20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]
        return mesh
    
    def comp_call_(self, cubefeats : SparseTensor, training=False):
        """
        Generates a mesh based on the specified sparse voxel structures.
        Args:
            cube_attrs [Nx21] : Sparse Tensor attrs about cube weights
            verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal 
        Returns:
            return the success tag and ni you loss, 
        """
        # add sdf bias to verts_attrs
        coords = cubefeats.coords[:, 1:]
        # print(coords) # torch.Size([484288, 3])
        # print(coords.shape)
        feats = cubefeats.feats
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
     
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True)
        idx_grid_3d, idx_grid_1d = build_idx_voxel_grid(v_pos, self.res)
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None

        # print(self.reg_v)
        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res) # [257**3, 3]存的是dense verts grid的所有vert deform后的坐标
        # print(x_nx3)
        # print(x_nx3.shape)
        # print(v_pos[100,...])

        # print(x_nx3.view(257,257,257,3)[v_pos[100,...][0],v_pos[100,...][1],v_pos[100,...][2]])


        

        tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res) #所有无重复的有效verts形变后的坐标（-0.5， 0.5） [570640, 3]
        tsdf_s = v_attrs[:, 0] #所有无重复的有效verts的sdf [570640]
        # print(v_pos / self.res - 0.5)
        # print(v_pos.shape) #torch.Size([570640, 3])
        # print(self.reg_v.shape) #[257**3, 3]
        # print(tsdf_v.shape) #torch.Size([570640, 3])
        
        # print(tsdf_v[idx_grid_3d[v_pos[100,...][0],v_pos[100,...][1],v_pos[100,...][2]]]) 
        # print(tsdf_s.shape) # torch.Size([570640])
        # print(tsdf_s)
        # print(idx_grid_3d)
        # print(idx_grid_3d.shape) # torch.Size([257, 257, 257])
        # print(idx_grid_1d)
        # print(idx_grid_1d.shape) # torch.Size([16974593])
        # print(aaa) 


        return sdf_d, x_nx3, tsdf_v, tsdf_s, idx_grid_3d, idx_grid_1d



def voxel_idx_to_points(voxel_idx, res, bbox_min=(-0.5,-0.5,-0.5), bbox_max=(0.5,0.5,0.5), order='zyx', centers=True):
    """
    voxel_idx: [N,3] int，默认顺序 z,y,x
    res: int，比如 256
    order: 若你的索引顺序是 x,y,z，就传 'xyz'
    centers: True 表示取体素中心；False 可改成顶点（+0）
    """
    voxel_idx = np.asarray(voxel_idx, dtype=np.int64)
    if order == 'xyz':
        x = voxel_idx[:,0]; y = voxel_idx[:,1]; z = voxel_idx[:,2]
    elif order == 'zyx':
        z = voxel_idx[:,0]; y = voxel_idx[:,1]; x = voxel_idx[:,2]
    else:
        raise ValueError("order 只能是 'zyx' 或 'xyz'")

    # 体素中心偏移 0.5；如果你想可视化体素“角点”或“顶点格(res+1)”，把 0.5 改成 0 或相应索引范围
    offset = 0.5 if centers else 0.0
    frac = np.stack([(z+offset)/res, (y+offset)/res, (x+offset)/res], axis=1)

    bbox_min = np.array(bbox_min, dtype=np.float32)
    bbox_max = np.array(bbox_max, dtype=np.float32)
    pts = bbox_min + frac * (bbox_max - bbox_min)  # [N,3] 世界坐标
    return pts.astype(np.float32)

def save_ply_xyz(path, points):
    points = np.asarray(points, dtype=np.float32)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        np.savetxt(f, points, fmt="%.6f")


def build_idx_voxel_grid(v_pos: torch.Tensor, res: int):
    """
    v_pos: (V,3) int 或 float，表示顶点的网格坐标（范围 0..res），因为xyz坐标都延展到了res+1
    返回:
      idx_grid_3d: (res+1, res+1, res+1) int32，位置存 v_pos 的行号(顶点 id)，无则为 -1
      idx_grid_1d: ((res+1)^3,) 同上，展平版
    """
    V = res + 1
    device = v_pos.device
    
    # print(v_pos)
    # print(v_pos.shape) #
    

    # 坐标 -> long，并夹到合法范围（可按需去掉 clamp）
    coords = v_pos.to(torch.long).clamp(0, V - 1)
    x, y, z = coords.unbind(-1)
    # print(x)
    # print(y)
    # print(z)
    
    # 线性索引
    lin = (x * V * V + y * V + z)              # (V,)
    ids = torch.arange(coords.shape[0], device=device, dtype=torch.int32)

    # 初始化为 -1
    idx_grid_1d = torch.full((V * V * V,), -1, dtype=torch.int32, device=device)

    # 按索引写入（重复位置时，后者覆盖前者）
    idx_grid_1d.index_put_((lin.to(torch.long),), ids, accumulate=False)
  
    # 还原 3D
    idx_grid_3d = idx_grid_1d.view(V, V, V)
    return idx_grid_3d, idx_grid_1d

import numpy as np
def save_ply(path, V, F):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
        np.savetxt(f, V, fmt="%.6f")
        np.savetxt(f, np.c_[np.full((len(F),1),3), F], fmt="%d")


