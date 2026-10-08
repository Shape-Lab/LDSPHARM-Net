"""
August 2022

Ilwoo Lyu, ilwoolyu@postech.ac.kr
Seunghwan Lee, shwan@postech.ac.kr

Medical Information Processing Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldspharmnet.lib.sphere import Icosphere


class CSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def loss_fn(self, query, point):
        cos_dist = torch.matmul(point, query)
        cos_dist = torch.clamp(cos_dist, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        return torch.sum(torch.acos(cos_dist))

    def forward(self, center, pole):
        # cosine similarity
        # range 0 ~ 2
        # cosloss = 1 - F.cosine_similarity(pole, center, dim=0)

        # cosine similarity + arccos
        # range 0 ~ pi
        cosloss = F.cosine_similarity(pole, center, dim=0)
        cosloss = torch.clamp(cosloss, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        cosloss = torch.acos(cosloss)
        return cosloss


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-12, weight=None, ignore_index=None, reduction="mean"):
        """
        Dice loss.

        Parameters
        __________
        eps : float
            Small value to avoid division by zero.
        weight : torch.tensor, shape=[n_class]
            A manual rescaling weight given to each class.
        ignore_index : list of int, shape=[n_ignore_classes]
            Target values that are ignored and do not contribute to the input gradient.
        reduction : str
            Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """

        super().__init__()

        self.eps = eps
        self.weight = weight if weight is not None else 1
        self.ignore_index = torch.tensor(ignore_index if ignore_index is not None else [], dtype=torch.long)

        if reduction == "mean":
            weight_sum = weight.sum() if weight is not None else 1
            self.reduction = lambda x: x.mean() / weight_sum
        elif reduction == "sum":
            self.reduction = torch.sum
        elif reduction == "none":
            self.reduction = nn.Identity()
        else:
            raise ValueError(f"reduction must be one of 'none', 'mean', or 'sum'. Got {reduction}.")

    def forward(self, input, target):
        input_soft = F.softmax(input, dim=1)
        target_onehot = torch.zeros_like(input_soft)
        target_onehot.scatter_(-2, target.unsqueeze(-2), 1)

        inter = (input_soft * target_onehot).sum(-1)
        denom = (input_soft + target_onehot).sum(-1).clamp_min(self.eps)
        loss = self.weight * (1 - (2 * inter / denom))

        mask = torch.ones_like(loss, dtype=torch.bool)
        mask[..., self.ignore_index] = False
        loss = loss[mask].reshape(*mask.shape[:-1], -1)
        loss = self.reduction(loss)

        return loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, eps=1e-12, weight=None, ignore_index=None, reduction="mean", alpha=0.3, beta=0.7, gamma=2.0):
        super().__init__()

        self.eps = eps
        self.weight = weight if weight is not None else 1
        self.ignore_index = torch.tensor(ignore_index if ignore_index is not None else [], dtype=torch.long)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if reduction == "mean":
            weight_sum = weight.sum() if weight is not None else 1
            self.reduction = lambda x: x.mean() / weight_sum
        elif reduction == "sum":
            self.reduction = torch.sum
        elif reduction == "none":
            self.reduction = nn.Identity()
        else:
            raise ValueError(f"reduction must be one of 'none', 'mean', or 'sum'. Got {reduction}.")

    def forward(self, input, target):
        input_soft = F.softmax(input, dim=1)
        target_onehot = torch.zeros_like(input_soft)
        target_onehot.scatter_(-2, target.unsqueeze(-2), 1)

        inter = (input_soft * target_onehot).sum(-1)

        FP = (input_soft * (1 - target_onehot)).sum(-1)
        FN = ((1 - input_soft) * target_onehot).sum(-1)

        tversky = inter / (inter + self.alpha * FP + self.beta * FN).clamp_min(self.eps)
        loss = self.weight * ((1 - tversky) ** self.gamma)

        mask = torch.ones_like(loss, dtype=torch.bool)
        mask[..., self.ignore_index] = False
        loss = loss[mask].reshape(*mask.shape[:-1], -1)
        loss = self.reduction(loss)

        return loss


class OmegaLoss(nn.Module):
    def __init__(self, device, gamma=500, delta=0.30):
        super().__init__()

        v, _ = Icosphere(6).sphere[6]
        self.v_gpu = torch.from_numpy(v).to(device=device, dtype=torch.float32)
        self.gamma = gamma
        self.delta = delta

    def forward(self, label, pole, angle):
        mask = label.squeeze(0) != 0
        coeff = mask.sum()

        dot = torch.matmul(self.v_gpu[mask, :], pole)
        geodesic = torch.arccos(torch.clamp(dot, min=-1.0 + 1e-7, max=1.0 - 1e-7))

        incentive = angle ** 2
        penalty = self.gamma / coeff * (F.relu(geodesic - angle + self.delta).sum())

        loss_omega = incentive + penalty
        vanish = ((geodesic - angle) > 0).sum()

        return loss_omega, vanish
