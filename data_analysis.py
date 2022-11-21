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
import matplotlib.pyplot as plt

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
        "image_name": images[idx].name,
    }


def depth_plot(image_dict):
    name = image_dict["image_name"]
    depths = np.array(image_dict["depth"])
    x_arr = range(len(depths[0]))
    avgs = []
    for x in x_arr:
        avgs.append(np.average(depths[:, x]))

    plt.plot(x_arr, avgs)
    plt.xlabel("X-coord")
    plt.ylabel("average depth")
    plt.savefig(f"depth_plot_{name}.png")
    print(f"Image being analysed is {name}")
    print(f"depth of size {depths.shape}")


# Added y-pos coloring and color bar
def depth_scatter_plot(image_dict):
    name = image_dict["image_name"]
    depths = np.array(image_dict["depth"])
    points = []
    for x in range(len(depths[0])):
        for y in range(len(depths)):
            points.append((x, depths[y][x], y))
    points = np.array(points)

    cbar = plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=1)
    plt.xlabel("X-coord")
    plt.ylabel("depth")
    plt.colorbar(cbar)
    plt.savefig(f"depth_scatter_plot_{name}.png")
    print(f"Image being analysed is {name}")
    print(f"depth of size {depths.shape}")


def reproj(image_dict):
    K_1 = image_dict["K"]
    depth_1 = image_dict["depth"]
    R1 = image_dict["R"]
    T1 = image_dict["T"]
    im = image_dict["image"]
    height = len(im)
    width = len(im[0])

    xs, ys = np.meshgrid(np.arange(0, height), np.arange(0, width))
    zs = np.ones((width, height))

    combined = np.stack((xs, ys, zs), axis=2)
    # x, y indexing
    combined_u = np.transpose(combined, axes=(1, 0, 2))
    # np meshgrid, each r,c (or c,r) is [x,y,z] vector
    # use that as u and apply whole transformation to it
    combined_u = np.expand_dims(combined_u, 3)
    p = np.linalg.solve(K_1, combined_u)

    combined_u = combined_u.astype(int)

    wp_est1 = np.array(R1) @ np.array(
        depth_1[combined_u[:, :, 0], combined_u[:, :, 1]][:, :, np.newaxis] * p
    )
    wp_est1 = wp_est1.squeeze()

    wp_est1 += T1
    return wp_est1


def combined_reprojection(image_dict_1, image_dict_2):
    im_1 = reproj(image_dict_1)
    im_2 = reproj(image_dict_2)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(im_1[:, :, 1], im_1[:, :, 0], s=1)
    ax1.scatter(im_2[:, :, 1], im_2[:, :, 0], s=1)
    plt.xlabel("X-coord")
    plt.ylabel("Y-coord")
    plt.savefig(f"reproj_scatter_plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image analysis tools.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="/datasets/reichstag/reichstag/dense",
        help="Path to the directory that contains the data",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="dump_match_pairs/",
        help="Path to the directory in which the .npz results and optionally,"
        "the visualization images are written",
    )

    parser.add_argument(
        "--image_idx",
        type=int,
        help="index of image to analyse.",
    )

    parser.add_argument(
        "--image_idx_2",
        type=int,
        help="index of image to analyse.",
    )

    opt = parser.parse_args()
    input_dir = Path(opt.input_dir)

    idx = opt.image_idx
    idx_2 = opt.image_idx_2
    image_dict = get_image(opt, str(input_dir), idx)
    # depth_plot(image_dict)
    # depth_scatter_plot(image_dict)
    combined_reprojection(
        get_image(opt, str(input_dir), idx), get_image(opt, str(input_dir), idx_2)
    )
