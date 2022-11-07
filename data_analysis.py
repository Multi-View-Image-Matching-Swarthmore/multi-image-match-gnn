from itertools import count
from pathlib import Path
import argparse
import random
import numpy as np
import torch


import numpy as np
from colmap.scripts.python.read_write_model import read_model, qvec2rotmat
from colmap.scripts.python.read_write_dense import read_array
from imageio.v2 import imread
import h5py

# import deepdish as dd
from time import time

from matplotlib import cm

from models.matching import Matching
from models.utils import (
    compute_pose_error,
    compute_epipolar_error,
    estimate_pose,
    make_matching_plot,
    error_colormap,
    AverageTimer,
    pose_auc,
    read_image,
    rotate_intrinsics,
    rotate_pose_inplane,
    scale_intrinsics,
    findDistance,
)

torch.set_grad_enabled(False)


def get_image(opt, src, idx):
    cameras, images, points = read_model(path=opt.input_dir + "/sparse", ext=".bin")
    im = imread(src + "/images/" + images[idx].name)
    depth = read_array(
        src + "/stereo/depth_maps/" + images[idx].name + ".photometric.bin"
    )
    min_depth, max_depth = np.percentile(depth, [5, 95])
    depth[depth < min_depth] = min_depth
    depth[depth > max_depth] = max_depth

    # reformat data
    q = images[idx].qvec
    R = qvec2rotmat(q)
    T = images[idx].tvec
    p = images[idx].xys
    pars = cameras[idx].params
    K = np.array([[pars[0], 0, pars[2]], [0, pars[1], pars[3]], [0, 0, 1]])
    pids = images[idx].point3D_ids
    v = pids >= 0
    print("Number of (valid) points: {}".format((pids > -1).sum()))
    print("Number of (total) points: {}".format(v.size))

    # get also the clean depth maps
    base = ".".join(images[idx].name.split(".")[:-1])
    with h5py.File(
        src + "/stereo/depth_maps_clean_300_th_0.10/" + base + ".h5", "r"
    ) as f:
        depth_clean = f["depth"][:]

    return {
        "image": im,
        "depth_raw": depth,
        "depth": depth_clean,
        "K": K,
        "q": q,
        "R": R,
        "T": T,
        "xys": p,
        "ids": pids,
        "valid": v,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image analysis tools.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/phototourism/reichstag/dense",
        help="Path to the directory that contains the data",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="dump_match_pairs/",
        help="Path to the directory in which the .npz results and optionally,"
        "the visualization images are written",
    )

    opt = parser.parse_args()
    # print(opt)
    input_dir = Path(opt.input_dir)
    # image_dir = input_dir / "images"
    # device = "cuda" if torch.cuda.is_available() and not opt.force_cpu else "cpu"
    # print('Running inference on device "{}"'.format(device))

    # pair = pair[0]
    # name0, name1 = pair[:2]
    # rot0, rot1 = 0, 0
    # image0, inp0, scales0 = read_image(
    #     image_dir / name0, device, opt.resize, rot0, opt.resize_float
    # )
    # image1, inp1, scales1 = read_image(
    #     image_dir / name1, device, opt.resize, rot1, opt.resize_float
    # )
    idx = 0
    image_dict = get_image(opt, str(input_dir), idx)
    print(image_dict)