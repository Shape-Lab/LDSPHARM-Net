"""
August 2022

Ilwoo Lyu, ilwoolyu@postech.ac.kr
Seungbo Ha, mj0829@unist.ac.kr
Seungeun Lee, selee@unist.ac.kr
Seunghwan Lee, shwan@postech.ac.kr

Medical Information Processing Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from spharmnet.core.models import SHConvBlock
from spharmnet.lib.sphere import vertex_area, spharm_real

from ldspharmnet.lib.sphere import Icosphere, TriangleSearchTorch, Adjacency, axis_angle_v, retess, arc_length


class LDSPHARM_Net(nn.Module):
    def __init__(self, device, in_ch=3, n_class=9, C=128, L=80, D=3, interval=5, epsilon=0.0, threads=1, method="exp", k=-0.5, verbose=False):
        super().__init__()
        v, f = Icosphere(6).sphere[6]
        v = v.astype(float)
        self.v_gpu = torch.from_numpy(v).type(torch.float32).to(device)

        area = vertex_area(v, f)
        area = torch.from_numpy(area).to(device=device, dtype=torch.float32)
        Y = spharm_real(v, L, threads)
        Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)

        self.adj = Adjacency(device)

        if verbose:
            print("Feature Extraction Block")
        self.extractor = extractBlock(Y, area, in_ch=in_ch, C=32, L=80, D=1, interval=interval, verbose=verbose)

        if verbose:
            print("Pole Estimation Block")
        self.pole_head = estimationBlock(Y, area, C=32, L=80, interval=interval, verbose=verbose)

        if verbose:
            print("Angle Estimation Block")
        self.angle_head = estimationBlock(Y, area, C=32, L=80, interval=interval, verbose=verbose)

        self.warp = largeWarpBlock(device, epsilon, method, k)

        if verbose:
            print("Sulcal Labeling Block")
        self.parc = SPHARM_Net(
            Y,
            area,
            in_ch=in_ch + 2,
            n_class=n_class,
            C=C,
            L=L,
            D=D,
            interval=interval,
            verbose=verbose,
        )
    
    def forward(self, x):
        extraction = self.extractor(x[:, :-1, :])

        pole = self.pole_head(extraction)
        pole = F.softmax(pole, dim=2)
        pole = F.normalize(pole.squeeze(0).squeeze(0) @ self.v_gpu, p=2, dim=0)

        angle = self.angle_head(extraction).squeeze(0).squeeze(0).mean(dim=0)
        angle = ((F.tanh(angle) / 2.5) + 0.6) * math.pi  # to prevent excessive warping with numerical instability [0.2pi, pi]
        
        x_retess, tree, fid, bary = self.warp(x, pole, angle)

        # input channel concatenate: we recommend to use below option
        # it makes model converge more faster
        bin_raw = (x_retess[0, -1, :] > 0).type(x_retess.dtype)  # assume first feature is always mean curvature
        # expand 1 ring
        bin = torch.hstack( (torch.any((bin_raw[self.adj.penta]) == 1, axis=1), torch.any(bin_raw[self.adj.hexa] == 1, axis=1)) ).type(x_retess.dtype)

        x_retess = torch.cat((x_retess[:, :-1, :], bin.unsqueeze(0).unsqueeze(0)), axis=1)

        # distance transform
        answer = bin.clone()
        for _ in range(30):
            stacked = ~torch.hstack( (torch.any((bin[self.adj.penta]) == 0, axis=1), torch.any(bin[self.adj.hexa] == 0, axis=1)) )
            answer = answer + stacked
            bin = stacked.clone()
        distance = (answer / 30).unsqueeze(0).unsqueeze(0)

        x_retess = torch.cat((x_retess, distance), axis=1)

        x_retess = self.parc(x_retess)  # SPHARM-Net

        return x_retess, pole, angle, tree, fid, bary

class extractBlock(nn.Module):
    def __init__(self, Y, area, in_ch=3, C=128, L=80, D=3, interval=5, verbose=False):
        super().__init__()

        L_in = L
        self.down = []
        self.up = []

        ch_inc = 2
        out_ch = C

        Y_inv = Y.T
        nonlinear = F.relu

        # encoding
        for i in range(D):
            L = L_in // 2**i
            if verbose:
                print("Down {}\t| C:{} -> {}\t| L:{}".format(i + 1, in_ch, out_ch, L))
            self.down.append(SHConvBlock(Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear, i == 0))
            in_ch = out_ch
            out_ch *= ch_inc

        L //= 2
        out_ch //= ch_inc
        in_ch = out_ch
        if verbose:
            print("Bottom\t| C:{} -> {}\t| L:{}".format(in_ch, out_ch, L))
        self.down.append(SHConvBlock(Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear, False))

        # decoding
        for i in range(D - 1):
            L = L_in // 2 ** (D - 1 - i)
            in_ch = out_ch * 2
            out_ch //= ch_inc
            if verbose:
                print("Up {}\t| C:{} -> {}\t| L:{}".format(i + 1, in_ch, out_ch, L))
            self.up.append(SHConvBlock(Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear, False))

        in_ch = out_ch * 2
        L *= 2
        if verbose:
            print("Up {}\t| C:{} -> {}\t| L:{}".format(D, in_ch, out_ch, L))
        self.up.append(SHConvBlock(Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear, True))

        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        x_ = [self.down[0](x)]
        for i in range(len(self.down) - 1):
            x_.append(self.down[i + 1](x_[-1]))
        x = x_[-1]
        for i in range(len(self.up)):
            x = torch.cat([x, x_[-2 - i]], dim=1)
            x = self.up[i](x)

        return x

class estimationBlock(nn.Module):
    def __init__(self, Y, area, C=128, L=80, interval=5, verbose=False):
        super().__init__()

        Y_inv = Y.T
        nonlinear = F.relu

        if verbose:
            print("Layer 1\t| C:{} -> {}\t| L:{}".format(C, C//2, L))
            print("Layer 2\t| C:{} -> {}\t| L:{}".format(C//2, 1, L))
        self.layer1 = SHConvBlock(Y, Y_inv, area, C, C//2, L, interval, nonlinear, True, True)
        self.layer2 = SHConvBlock(Y, Y_inv, area, C//2, 1, L, interval, None, True, False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class SPHARM_Net(nn.Module):
    def __init__(self, Y, area, in_ch=3, n_class=32, C=128, L=80, D=3, interval=5, verbose=False):
        """
        SPHARM-Net [1].
        In the encoding (decoding) phase, each block halves (doubles) the harmonic bandwidth L while doubling (halving)
        the number of channels C. The final block aggregates the learned information to infer parcellation labels.
        The SHConv block performs SHT on a spherical signal followed by rotation-equivariant spectral convolution and ISHT.
        The SHConv-FB block adds a scaled spherical signal to the SHConv block to support constrained full-bandwidth convolution.

        [1] Ha, Seungbo, and Ilwoo Lyu.
            "SPHARM-Net: Spherical Harmonics-based Convolution for Cortical Parcellation."
            IEEE Transactions on Medical Imaging 41, no. 10 (2022): 2739-2751.

        Parameters
        __________
        sphere : str
            Sphere mesh file. VTK (ASCII) or FreeSurfer format.
        device : torch.device
            Device indicator.
        in_ch : int
            # of input geometric features.
        n_class : int
            # of labels, i.e., the output layer size.
        C : int
            # of channels in the entry layer (see the paper for details).
        L : int
            Spectral bandwidth that supports individual component learning (see the paper for details).
        D : int
            Depth of encoding/decoding levels (see the paper for details).
        interval : int
            Interval of anchor points (see the paper for details).
        threads : int
            # of CPU threads for basis reconstruction. Useful if the unit sphere has dense tesselation.

        Notes
        _____
        In channel shape  : [batch, in_ch, n_vertex]
        Out channel shape : [batch, n_class, n_vertex]
        """

        super().__init__()

        L_in = L
        self.down = []
        self.up = []

        ch_inc = 2
        out_ch = C

        Y_inv = Y.T
        nonlinear = F.relu

        # encoding
        for i in range(D):
            L = L_in // 2**i
            if verbose:
                print("Down {}\t| C:{} -> {}\t| L:{}".format(i + 1, in_ch, out_ch, L))
            self.down.append(SHConvBlock(Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear, i == 0))
            in_ch = out_ch
            out_ch *= ch_inc

        L //= 2
        out_ch //= ch_inc
        in_ch = out_ch
        if verbose:
            print("Bottom\t| C:{} -> {}\t| L:{}".format(in_ch, out_ch, L))
        self.down.append(SHConvBlock(Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear, False))

        # decoding
        for i in range(D - 1):
            L = L_in // 2 ** (D - 1 - i)
            in_ch = out_ch * 2
            out_ch //= ch_inc
            if verbose:
                print("Up {}\t| C:{} -> {}\t| L:{}".format(i + 1, in_ch, out_ch, L))
            self.up.append(SHConvBlock(Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear, False))

        in_ch = out_ch * 2
        L *= 2
        if verbose:
            print("Up {}\t| C:{} -> {}\t| L:{}".format(D, in_ch, out_ch, L))
        self.up.append(SHConvBlock(Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear, True))
        if verbose:
            print("Final\t| C:{} -> {}\t| L:{}".format(out_ch, n_class, L))
        self.final = SHConvBlock(Y, Y_inv, area, out_ch, n_class, L_in, interval, None, True)

        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        x_ = [self.down[0](x)]
        for i in range(len(self.down) - 1):
            x_.append(self.down[i + 1](x_[-1]))
        x = x_[-1]
        for i in range(len(self.up)):
            x = torch.cat([x, x_[-2 - i]], dim=1)
            x = self.up[i](x)
        x = self.final(x)

        return x

class largeWarpBlock(nn.Module):
    def __init__(self, device, epsilon, method, k):
        super().__init__()

        v, f = Icosphere(6).sphere[6]
        v = v.astype(float)
        self.v_gpu = torch.from_numpy(v).type(torch.float32).to(device)
        self.f_gpu = torch.from_numpy(f).type(torch.int64).to(device)

        self.epsilon = epsilon
        self.method = method
        self.k = k

    def forward(self, input, pole, angle):
        dot = torch.matmul(self.v_gpu, pole)
        dot = arc_length(dot)

        mask = dot < angle
        if self.method == "linear":
            dot[mask] = (math.pi - angle) / angle * dot[mask] * (1 - self.epsilon)
        elif self.method == "sin":
            dot[mask] = (math.pi - angle) * torch.sin(math.pi / 2 / angle * dot[mask]) * (1 - self.epsilon)
        elif self.method == "exp":
            dot[mask] = (math.pi - angle) * (torch.exp(self.k * dot[mask]) - 1) / (torch.exp(self.k * angle) - 1) * (1 - self.epsilon)
        dot[~mask] = (math.pi - dot[~mask]) * (1 - self.epsilon)

        pole_expanded = pole.unsqueeze(0).expand_as(self.v_gpu)
        pole_expanded = torch.linalg.cross(pole_expanded, self.v_gpu)
        pole_expanded = F.normalize(pole_expanded, p=2, dim=1)

        rotated = axis_angle_v(pole_expanded, dot.unsqueeze(1), self.v_gpu)

        tree = TriangleSearchTorch(rotated, self.f_gpu)
        x_retess, fid, bary = retess(self.v_gpu.unsqueeze(0), input, tree)

        return x_retess, tree, fid, bary
