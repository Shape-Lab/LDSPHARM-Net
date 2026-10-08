"""
July 2021

Ilwoo Lyu, ilwoolyu@postech.ac.kr
Seungbo Ha, mj0829@unist.ac.kr
Seunghwan Lee, shwan@postech.ac.kr

Medical Information Processing Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import numpy as np
from scipy.sparse import csr_array

import torch
import torch.nn.functional as F

import TriangleSearchCUDA


class TriangleSearchTorch:
    def __init__(self, v, f, norm=False):
        """
        GPU-accelerated fast closest triangle search.
        This module only works on spherical tessellation.
        If a GPU device is unavailable, TriangleSearch will be used instead.

        Parameters
        __________
        v : torch.tensor, shape = [n_vertex, 3]
            3D coordinates of the unit sphere.
        f : torch.tensor, shape = [n_face, 3]
            Triangles of the unit sphere.
        norm : bool
            Enforce unit sphere.
        """

        self.v = v
        self.f = f

        if norm:
            self.v = self.v / torch.norm(self.v, dim=1, keepdim=True)

        a = self.v[self.f[:, 0], :]
        b = self.v[self.f[:, 1], :]
        c = self.v[self.f[:, 2], :]

        normal = torch.linalg.cross(b - a, c - b, dim=-1)
        area = torch.norm(normal, dim=1, keepdim=True)
        zero_normal = area < 1e-8
        normal = torch.where(zero_normal.expand_as(normal), a, normal)
        area = torch.where(zero_normal, torch.ones_like(area), area)
        normal = normal / area

        self.inner = (normal * a).sum(dim=1, keepdim=True)
        self.normal = normal
        self.area = area

        self.tree = TriangleSearchCUDA
        self.offset = torch.Tensor()
        self.cand = torch.Tensor()
        self.ncand = self.f.shape[0]

    def barycentric(self, v, f, q, fid, area, normal):
        """
        Barycentric coefficients of q inside triangles f.
        """

        n = normal[fid]
        area = area[fid]
        qa = v[f[fid, 0], :] - q
        qb = v[f[fid, 1], :] - q
        qc = v[f[fid, 2], :] - q

        u = (torch.linalg.cross(qb, qc) * n).sum(dim=1, keepdim=True) / area
        v = (torch.linalg.cross(qc, qa) * n).sum(dim=1, keepdim=True) / area
        w = 1 - u - v

        return torch.hstack((u, v, w))

    def query(self, q, tol=1e-4, norm=False, bary=True, cand=None, ncand=None, offset=None):
        """
        Find triangles that contain query points and then compute their barycentric coefficients.
        """

        dim = q.shape
        q = q.reshape(-1, 3)
        device = q.device

        if norm:
            q = q / torch.norm(q, dim=1, keepdim=True)

        if device.type != "cpu":
            if offset is None:
                offset = self.offset.to(device)
            if cand is None:
                cand = self.cand.to(device)
            if ncand is None:
                ncand = self.ncand
            fid = self.tree.query(self.v, self.f, cand, ncand, offset, q, self.normal, self.inner, self.area, tol)
        else:
            fid, _ = self.tree.query(q.detach().numpy(), tol=tol)
            fid = torch.from_numpy(fid).to(device)

        if bary:
            q = q * (self.inner[fid] / (q * self.normal[fid]).sum(dim=1, keepdim=True))
            b = self.barycentric(self.v, self.f, q, fid, self.area, self.normal)
            fid = fid.reshape(dim[:-1])
            b = b.reshape(dim)
            return fid, b
        else:
            fid = fid.reshape(dim[:-1])
            return fid

def axis_angle_v(axis, angle, v):
    """
    vertex-wise rotation by axis and angle.

    Parameters
    __________
    axis : torch.tensor, shape = [*, n_vertex, 3]
        Rotation axis.
    angle : torch.tensor, shape = [*, n_vertex, 1]
        Rotation angle.
    v : torch.tensor, shape = [*, n_vertex, 3]
        3D coordinates of the unit sphere.

    Returns
    _______
    x : torch.tensor, shape = [*, n_vertex, 3]
        3D warped coordinates of the unit sphere.
    """
    dot = (axis * v).sum(dim=-1, keepdim=True)
    # cross = torch.linalg.cross(axis, v[None, ...])
    cross = torch.linalg.cross(axis, v)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    x = v * cos + cross * sin + axis * (dot * (1 - cos))
    x = F.normalize(x, dim=-1)

    return x

def retess(x, feat, tree):
    """
    Data retessellation. The resulting geometry is sampeld at x.

    Parameters
    __________
    x : torch.tensor, shape = [batch, n_vertex_x, 3]
        3D coordinates of the unit sphere.
    feat : torch.tensor, shape = [batch, in_ch, n_vertex_feat]
        Geometric features.
    tree : TriangleSearchTorch or its variants
        Triangle search tree for feat.

    Returns
    _______
    tess : torch.tensor, shape = [batch, in_ch, n_vertex_x]
        Retessellated geometric features.
    """
    fid, bary = tree.query(x)

    fid = tree.f[fid]

    fid_feat = torch.repeat_interleave(fid[:, None], feat.shape[1], dim=1).flatten(-2, -1)
    bary_feat = torch.repeat_interleave(bary[:, None], feat.shape[1], dim=1)

    tess = (torch.gather(feat, -1, fid_feat).reshape(bary_feat.shape) * bary_feat).sum(dim=-1)

    return tess, fid, bary

class Icosphere:
    def __init__(self, level):
        """
        Vectorized icospheres derived from an icosahedron.
        The module returns a list of vertices and faces.

        Parameters
        __________
        level : int
            The level of recursion to build an icosphere.
        """

        self.sphere = [self.icosahedron()]
        for i in range(level):
            self.sphere.append(self.subdivision(*self.sphere[i]))

    def icosahedron(self):
        v = np.array(
            [
                [0.00000000000, 0.00000000000, 1.00000000000],
                [0.89440002441, 0.00000000000, 0.44720001221],
                [0.27639999390, 0.85069999695, 0.44720001221],
                [-0.72360000610, 0.52569995880, 0.44720001221],
                [-0.72360000610, -0.52569995880, 0.44720001221],
                [0.27639999390, -0.85069999695, 0.44720001221],
                [0.72360000610, -0.52569995880, -0.44720001221],
                [0.72360000610, 0.52569995880, -0.44720001221],
                [-0.27639999390, 0.85069999695, -0.44720001221],
                [-0.89440002441, 0.00000000000, -0.44720001221],
                [-0.27639999390, -0.85069999695, -0.44720001221],
                [0.00000000000, 0.00000000000, -1.00000000000],
            ]
        )
        f = np.array(
            [
                [0, 3, 4],
                [0, 4, 5],
                [0, 5, 1],
                [0, 1, 2],
                [0, 2, 3],
                [3, 2, 8],
                [3, 8, 9],
                [3, 9, 4],
                [4, 9, 10],
                [4, 10, 5],
                [5, 10, 6],
                [5, 6, 1],
                [1, 6, 7],
                [1, 7, 2],
                [2, 7, 8],
                [8, 11, 9],
                [9, 11, 10],
                [10, 11, 6],
                [6, 11, 7],
                [7, 11, 8],
            ],
            dtype=int,
        )
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        return v, f

    def edge(self, v, f):
        idx = f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).T

        # unique ordered edges
        tf = idx[0, :] < idx[1, :]
        order0 = np.flatnonzero(tf)
        order1 = np.flatnonzero(~tf)
        order0 = csr_array((order0, idx[:, order0]), shape=(v.shape[0], v.shape[0]))[(*idx[:, order1][[1, 0], :],)]
        sel = np.zeros(idx.shape[1], dtype=bool)
        sel[np.where(order0 < order1, order0, order1)] = True
        idx = idx[:, sel]

        return idx

    def subdivision(self, v, f):
        # edges
        idx = self.edge(v, f)

        # midpoints
        vNew = v[idx[0, :], :] + v[idx[1, :], :]
        vNew /= np.linalg.norm(vNew, axis=1, keepdims=True)
        vNew = np.vstack((v, vNew))

        # lookup table
        idx = csr_array(
            (
                np.tile(np.arange(v.shape[0], idx.shape[1] + v.shape[0]), 2),
                (idx.flatten(), idx[[1, 0], :].flatten()),
            ),
            shape=(v.shape[0], v.shape[0]),
        )

        # midpoint indices
        vID1 = idx[f[:, 0], f[:, 1]]
        vID2 = idx[f[:, 1], f[:, 2]]
        vID3 = idx[f[:, 2], f[:, 0]]

        # new triangles
        fNew = np.empty((3, f.shape[0] * 4), dtype=int)
        fNew[:, 0::4] = [f[:, 0], vID1, vID3]
        fNew[:, 1::4] = [vID1, f[:, 1], vID2]
        fNew[:, 2::4] = [vID3, vID2, f[:, 2]]
        fNew[:, 3::4] = [vID1, vID2, vID3]
        fNew = np.ascontiguousarray(fNew.T)

        return vNew, fNew

class Adjacency:
    def __init__(self, device):
        v, f = Icosphere(6).sphere[6]  # float64, int64
        v = v.astype(float) # explicit casting
        f = torch.from_numpy(f)
        stack_torch = torch.cat((torch.stack((f[:, 0], f[:, 1]), dim=1), torch.stack((f[:, 1], f[:, 2]), dim=1), torch.stack((f[:, 2], f[:, 0]), dim=1)), dim=0).to(device)
        query = stack_torch[:, 0]
        response = stack_torch[:, 1]
        self.penta = torch.zeros((12, 5), dtype=int, device=device)
        self.hexa = torch.zeros((v.shape[0] - 12, 6), dtype=int, device=device)
        for q in range(12):
            cand = query == q
            self.penta[q, :] = response[cand]
        for q in range(v.shape[0] - 12):
            cand = query == (q + 12)
            self.hexa[q, :] = response[cand]

def arc_length(data, eps=1e-7):
    return torch.arccos(torch.clamp(data, min=-1.0 + eps, max=1.0 - eps))

def label_retess(label, fid, bary):
    label = label[:, fid.gather(2, torch.argmax(bary, axis=2).unsqueeze(-1)).squeeze()]
    return label
