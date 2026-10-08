"""
Example script for sulcal labeling.

If you use this code, please cite the following paper.

    Seunghwan Lee, Jiwon Son, Seungeun Lee, Ethan H. Willbrand, Benjamin J. Parker, Silvia A. Bunge, Kevin S. Weiner, and Ilwoo Lyu
    Isotropic spherical region enlargement and background contraction for sulcal labeling
    Medical Image Analysis, vol. 115, 104303, 2027

    Seunghwan Lee, Jiwon Son, Seungeun Lee, Ethan H. Willbrand, Benjamin J. Parker, Kevin S. Weiner, and Ilwoo Lyu
    Extensive Spherical Region Enlargement With Isotropic Deformation for Sulcal Labeling
    2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI), pp. 1-5

Copyright 2026 Seunghwan Lee

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import math

from ldspharmnet import LDSPHARM_Net
from ldspharmnet.lib.utils import normalize_data, squeeze_label, refine_label
from ldspharmnet.lib.sphere import TriangleSearchTorch, Icosphere, axis_angle_v, arc_length
from ldspharmnet.lib.io import read_feat, read_mesh, write_vtk


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subj-dir", type=str, help="Path to FreeSurfer's surf", required=True)
    parser.add_argument("--feat-dir", type=str, default="surf", help="Path to geometry for registration metrics")
    parser.add_argument("--native-sphere-dir", type=str, default="surf", help="Path to native sphere")
    parser.add_argument("--out-to-subj", action="store_true", help="Set output's root as the subject dir")
    parser.add_argument("--out-dir", type=str, default="./output", help="Path to inferred labels (output)")
    parser.add_argument("--surface", type=str, default=None, help="Native surface mesh (white, inflated, etc.)")
    parser.add_argument("--native-sphere", type=str, default="sphere", help="Native sphere mesh (sphere, sphere.reg, etc.)")
    parser.add_argument("--ckpt", type=str, default="./logs/best_model_fold1.pth", help="Trained model for inference")
    parser.add_argument("--hemi", type=str, choices=["lh", "rh"], help="Hemisphere for inference", required=True)

    parser.add_argument("--no-cc", action="store_true", help="Do not find the largest components as post-processing.")

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for inference")
    parser.add_argument("--threads", type=int, default=1, help="# of CPU threads for basis reconstruction")
    parser.add_argument("--det", action="store_true", help="Enable deterministic algorithms for reproducibility")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data shuffling")

    args = parser.parse_args()

    return args


def read_subj(subj_dir, feat_dir, hemi, in_ch, data_norm):
    feat_dir = os.path.join(subj_dir, feat_dir)

    feat_list = []
    for ch in in_ch:
        f = os.path.join(feat_dir, hemi + "." + ch)
        feat_list.append(f)

    data = np.array([])
    for f in feat_list:
        feat = read_feat(f)
        data = np.append(data, feat)
    data = np.reshape(data, (len(feat_list), -1)).astype(np.float32)

    curv = data[0, :].copy()
    if data_norm:
        data = normalize_data(data)
    data = np.vstack((data, curv))  # 4, 40962

    return data


def main(args):
    device = torch.device(f"cuda:{args.gpu}")

    torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.det:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        print("Using deterministic algorithms.")

    ico_v, ico_f = Icosphere(6).sphere[6]

    print("Checkpoint: ", args.ckpt)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]
    if ckpt_args["in_ch"][-1] == "iH":
        ckpt_args["in_ch"][-1] = "inflated.H"
    _, lut = squeeze_label(ckpt_args["classes"])

    print("Loading data...")
    native_data = torch.from_numpy(read_subj(args.subj_dir, args.feat_dir, args.hemi, ckpt_args["in_ch"], ckpt_args["data_norm"])).to(device=device, dtype=torch.float)
    sphere = os.path.join(args.subj_dir, "surf", args.hemi + "." + args.native_sphere)

    native_v, native_f = read_mesh(sphere)
    native_v = torch.from_numpy(native_v).to(device=device, dtype=torch.float)
    native_f = torch.from_numpy(native_f).to(device=device, dtype=torch.long)
    ico_v = torch.from_numpy(ico_v).to(device=device, dtype=torch.float)
    ico_f = torch.from_numpy(ico_f).to(device=device, dtype=torch.long)
    native_v = F.normalize(native_v, p=2, dim=1)

    tree = TriangleSearchTorch(native_v, native_f)
    triangle_idx, bary_coeff = tree.query(ico_v)
    in_data = torch.multiply(native_data[:, native_f[triangle_idx]], bary_coeff).sum(-1).unsqueeze(0)

    print("Loading model...")
    model = LDSPHARM_Net(
        device=device,
        in_ch=len(ckpt_args["in_ch"]),
        n_class=len(ckpt_args["classes"]),
        C=ckpt_args["channel"],
        L=ckpt_args["bandwidth"],
        D=ckpt_args["depth"],
        interval=ckpt_args["interval"],
        epsilon=ckpt_args["epsilon"],
        threads=args.threads,
        method=ckpt_args["method"],
        k=ckpt_args["k"],
        verbose=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Making inference...")
    with torch.no_grad():
        output, pole, angle, _, _, _ = model(in_data)
    output = torch.squeeze(output)

    tree = TriangleSearchTorch(ico_v, ico_f)
    # large deform
    dot = torch.matmul(native_v, pole)
    dot = arc_length(dot)

    mask = dot < angle
    if ckpt_args["method"] == "linear":
        dot[mask] = (math.pi - angle) / angle * dot[mask] * (1 - ckpt_args["epsilon"])
    elif ckpt_args["method"] == "sin":
        dot[mask] = (math.pi - angle) * torch.sin(math.pi / 2 / angle * dot[mask]) * (1 - ckpt_args["epsilon"])
    elif ckpt_args["method"] == "exp":
        dot[mask] = (math.pi - angle) * (torch.exp(ckpt_args["k"] * dot[mask]) - 1) / (torch.exp(ckpt_args["k"] * angle) - 1) * (1 - ckpt_args["epsilon"])
    dot[~mask] = (math.pi - dot[~mask]) * (1 - ckpt_args["epsilon"])

    pole_expanded = pole.unsqueeze(0).expand_as(native_v)
    pole_expanded = torch.linalg.cross(pole_expanded, native_v)
    pole_expanded = F.normalize(pole_expanded, p=2, dim=1)

    rotated_native = axis_angle_v(pole_expanded, dot.unsqueeze(1), native_v)

    triangle_idx, bary_coeff = tree.query(rotated_native)
    output = torch.multiply(output[:, ico_f[triangle_idx]], bary_coeff).sum(-1)
    output = output.cpu().detach().numpy()

    # This is not a part of the publication but generally provides clean connected components.
    # Do NOT enable this option if any ROI has more than one connected component, e.g., sulcus
    if not args.no_cc:
        print("Refining labels...")
        output = refine_label(output, native_f)

    parc = np.argmax(output, axis=0)
    parc = [lut[p] for p in parc]

    out_dir = args.out_dir
    if args.out_to_subj:
        out_dir = os.path.join(args.subj_dir, out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Saving labels...")
    if args.out_to_subj:
        out_file = args.hemi + ".label.txt"
    else:
        out_prefix = list(filter(None, args.subj_dir.split("/")))
        out_prefix = out_prefix[-1] + "." + args.hemi if len(out_prefix) > 1 else args.hemi
        out_file = out_prefix + ".label.txt"
    np.savetxt(os.path.join(out_dir, out_file), parc, delimiter=",", fmt="%d")
    
    # if args.surface is not None:
    #     if args.out_to_subj:
    #         out_file = args.hemi + ".label.vtk"
    #     else:
    #         out_file = out_prefix + ".label.vtk"
    #         surf_v, surf_f = read_mesh(os.path.join(args.subj_dir, args.feat_dir, args.hemi + "." + args.surface))
    #     write_vtk(os.path.join(out_dir, out_file), surf_v, surf_f, {"label": parc})


if __name__ == "__main__":
    args = get_args()
    main(args)
