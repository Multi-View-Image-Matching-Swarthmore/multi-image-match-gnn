#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

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

from match_pair_helpers import get_image, pairwise_match

torch.set_grad_enabled(False)

# This function should be defined in (and imported from) models.utils as above


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image pair matching and pose evaluation with SuperGlue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_pairs",
        type=str,
        default="assets/reichstag_image_pairs.txt",
        help="Path to the list of image pairs",
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

    parser.add_argument(
        "--max_length",
        type=int,
        default=-1,  # changed from -1 to 1
        help="Maximum number of pairs to evaluate",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs="+",
        default=[640, 480],
        help="Resize the input image before running inference. If two numbers, "
        "resize to the exact dimensions, if one number, resize the max "
        "dimension, if -1, do not resize",
    )
    parser.add_argument(
        "--resize_float",
        action="store_true",
        help="Resize the image after casting uint8 to float",
    )

    parser.add_argument(
        "--superglue",
        choices={"indoor", "outdoor"},
        default="outdoor",
        help="SuperGlue weights",
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=1024,
        help="Maximum number of keypoints detected by Superpoint"
        " ('-1' keeps all keypoints)",
    )
    parser.add_argument(
        "--keypoint_threshold",
        type=float,
        default=0.005,
        help="SuperPoint keypoint detector confidence threshold",
    )
    parser.add_argument(
        "--nms_radius",
        type=int,
        default=4,
        help="SuperPoint Non Maximum Suppression (NMS) radius" " (Must be positive)",
    )
    parser.add_argument(
        "--sinkhorn_iterations",
        type=int,
        default=20,
        help="Number of Sinkhorn iterations performed by SuperGlue",
    )
    parser.add_argument(
        "--match_threshold", type=float, default=0.2, help="SuperGlue match threshold"
    )

    parser.add_argument(
        "--viz", action="store_true", help="Visualize the matches and dump the plots"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Perform the evaluation" " (requires ground truth pose and intrinsics)",
    )
    parser.add_argument(
        "--fast_viz",
        action="store_true",
        help="Use faster image visualization with OpenCV instead of Matplotlib",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Skip the pair if output .npz files are already found",
    )
    parser.add_argument(
        "--show_keypoints",
        action="store_true",
        help="Plot the keypoints in addition to the matches",
    )
    parser.add_argument(
        "--viz_extension",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Visualization file extension. Use pdf for highest-quality.",
    )
    parser.add_argument(
        "--opencv_display",
        action="store_true",
        help="Visualize via OpenCV before saving output images",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle ordering of pairs before processing",
    )
    parser.add_argument(
        "--force_cpu", action="store_true", help="Force pytorch to run in CPU mode."
    )

    opt = parser.parse_args()
    print(opt)

    assert not (
        opt.opencv_display and not opt.viz
    ), "Must use --viz with --opencv_display"

    assert not (
        opt.opencv_display and not opt.fast_viz
    ), "Cannot use --opencv_display without --fast_viz"

    assert not (opt.fast_viz and not opt.viz), "Must use --viz with --fast_viz"

    assert not (
        opt.fast_viz and opt.viz_extension == "pdf"
    ), "Cannot use pdf extension with --fast_viz"

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print("Will resize to {}x{} (WxH)".format(opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print("Will resize max dimension to {}".format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print("Will not resize images")
    else:
        raise ValueError("Cannot specify more than two integers for --resize")

    with open(opt.input_pairs, "r") as f:
        pairs = [l.split() for l in f.readlines()]

    if opt.max_length > -1:
        pairs = pairs[0 : np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    if opt.eval:
        if not all([len(p) == 38 for p in pairs]):
            raise ValueError(
                "All pairs should have ground truth info for evaluation."
                'File "{}" needs 38 valid entries per row'.format(opt.input_pairs)
            )
    pair = pairs[:2]
    pairwise_match(opt, pair)