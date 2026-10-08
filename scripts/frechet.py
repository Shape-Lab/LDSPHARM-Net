"""
Example script for estimating Fréchet mean.

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

from ldspharmnet.lib.sphere import Icosphere
from ldspharmnet.lib.loss import CSLoss
from ldspharmnet.lib.io import read_mesh, read_dat


def get_args():
    parser = argparse.ArgumentParser()

    # Dataset & dataloader
    parser.add_argument("--data-dir", type=str, default="./dataset", help="Path to re-tessellated data")
    parser.add_argument("--classes", type=int, nargs="+", help="List of regions of interest")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data shuffling")
    parser.add_argument("--out-dir", type=str, default="./frechet", help="Path to inferred labels (output)")

    # Training and evaluation
    parser.add_argument("--epochs", type=int, default=200, help="Max epoch")
    
    # Machine settings
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for training (normally, starting with 0)")
    parser.add_argument("--no-cuda", action="store_true", help="No CUDA")
    parser.add_argument("--det", action="store_true", help="Enable deterministic algorithms for reproducibility")

    args = parser.parse_args()

    return args


def frechet(point, device, epochs, MAX=2100000):
    query = torch.tensor([point[0, 0], point[0, 1], point[0, 2]], requires_grad=True, device=device)
    optimizer = torch.optim.SGD([query], lr=0.01)
    criterion = CSLoss.loss_fn
    best = MAX

    result = -1
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(_, query, point)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            query.data = query / query.norm()
        if best > loss:
            best = loss
            result = query
    return result.detach().cpu().numpy()

def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")

    torch.set_num_threads(1)

    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.det:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        print("Using deterministic algorithms.")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    v, _ = Icosphere(6).sphere[6]
    v_gpu = torch.from_numpy(v).type(torch.float32).to(device)

    # dataset
    print("Loading data list...")
    subjs = os.listdir(args.data_dir + "/labels")
    
    # Fréchet mean of labels
    print("Calculating Fréchet mean of labels...")
    for subj in subjs:
        print("Processing", subj)
        splitted = subj.split('.')
        splitted[-2] = 'frechet'
        splitted[-1] = 'txt'

        src_pth = args.data_dir + "/labels/" + subj
        dst_pth = args.out_dir + "/" + '.'.join(splitted)
        label = torch.from_numpy(read_dat(src_pth)).type(torch.int16).to(device)

        centers = []
        # all ROIs
        mask = label != 0
        point = v_gpu[mask, :]
        centers.append(frechet(point, device, args.epochs))
        
        # # ROI-wise
        # for i in args.classes[1:]:
        #     mask = label == i
        #     point = v_gpu[mask, :]
        #     centers.append(frechet(point, device, args.epochs))
        
        centers = np.array(centers).reshape(-1)
        np.savetxt(dst_pth, centers, fmt="%f")

if __name__ == "__main__":
    args = get_args()
    main(args)
