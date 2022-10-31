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


def get_homogenous_coords(pt):
    return np.array([pt[1], pt[0], 1])


def calculate_distance(pair, mkpts0, mkpts1, data_one, data_two):
    u, v = [], []
    K_1 = data_one["K"]
    K_2 = data_two["K"]

    # print(pair)
    depth_1 = data_one["depth"]
    depth_2 = data_two["depth"]

    R1 = data_one["R"].T
    R2 = data_two["R"].T

    T1 = -R1 @ data_one["T"]  # -R1 @
    T2 = -R2 @ data_two["T"]  # -R2 @
    print("T1:")
    print(T1)

    u = get_homogenous_coords(mkpts0)
    v = get_homogenous_coords(mkpts1)
    # print(u)
    # print(v)
    p = np.linalg.solve(K_1, u)
    q = np.linalg.solve(K_2, v)
    wp_est1 = np.array(R1) @ np.array(depth_1[int(u[0]), int(u[1])] * p)
    wp_est1 += T1  # issue
    wp_est2 = np.array(R2) @ np.array(depth_2[int(v[0]), int(v[1])] * q)
    wp_est2 += T2
    error = np.linalg.norm((wp_est1 - wp_est2))

    return error
    print(wp_est1)
    print(wp_est2)
    print(error)


def reprojection_error(pair, mkpts0, mkpts1, data_one, data_two):
    u, v = [], []
    K_1 = data_one["K"]
    K_2 = data_two["K"]

    # print(pair)
    depth_1 = data_one["depth"]
    depth_2 = data_two["depth"]

    R1 = data_one["R"].T
    R2 = data_two["R"].T

    T1 = -R1 @ data_one["T"]  # -R1 @
    T2 = -R2 @ data_two["T"]  # -R2 @

    u = get_homogenous_coords(mkpts0)
    v = get_homogenous_coords(mkpts1)
    # print(u)
    # print(v)
    p = np.linalg.solve(K_1, u)
    q = np.linalg.solve(K_2, v)
    wp_est1 = np.array(R1) @ np.array(depth_1[int(u[0]), int(u[1])] * p)
    wp_est1 += T1
    point1_in_f2 = (np.array(R2) @ wp_est1) - T2
    # wp_est2 = np.array(R2) @ np.array(depth_2[v[0], v[1]] * q)
    # wp_est2 += T2
    error = np.linalg.norm((point1_in_f2 - np.array(depth_2[int(v[0]), int(v[1])] * q)))

    return error


def pairwise_match(opt, pair):
    # Load models
    device = "cuda" if torch.cuda.is_available() and not opt.force_cpu else "cpu"
    print('Running inference on device "{}"'.format(device))
    config = {
        "superpoint": {
            "nms_radius": opt.nms_radius,
            "keypoint_threshold": opt.keypoint_threshold,
            "max_keypoints": opt.max_keypoints,
        },
        "superglue": {
            "weights": opt.superglue,
            "sinkhorn_iterations": opt.sinkhorn_iterations,
            "match_threshold": opt.match_threshold,
        },
    }
    matching = Matching(config).eval().to(device)

    # Setting up for matching
    timer = AverageTimer(newline=True)
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    input_dir = Path(opt.input_dir)
    image_dir = input_dir / "images"

    # for i, pair in enumerate(pairs):
    pair = pair[0]
    name0, name1 = pair[:2]
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    matches_path = output_dir / "{}_{}_matches.npz".format(stem0, stem1)
    eval_path = output_dir / "{}_{}_evaluation.npz".format(stem0, stem1)
    viz_path = output_dir / "{}_{}_matches.{}".format(stem0, stem1, opt.viz_extension)
    viz_eval_path = output_dir / "{}_{}_evaluation.{}".format(
        stem0, stem1, opt.viz_extension
    )

    # Handle --cache logic.
    do_match = True
    do_eval = opt.eval
    do_viz = opt.viz
    do_viz_eval = opt.eval and opt.viz

    if opt.cache:
        if matches_path.exists():
            try:
                results = np.load(matches_path)
            except:
                raise IOError("Cannot load matches .npz file: %s" % matches_path)

            kpts0, kpts1 = results["keypoints0"], results["keypoints1"]
            matches, conf = results["matches"], results["match_confidence"]
            do_match = False
        if opt.eval and eval_path.exists():
            try:
                results = np.load(eval_path)
            except:
                raise IOError("Cannot load eval .npz file: %s" % eval_path)
            err_R, err_t = results["error_R"], results["error_t"]
            precision = results["precision"]
            matching_score = results["matching_score"]
            num_correct = results["num_correct"]
            epi_errs = results["epipolar_errors"]
            do_eval = False
        if opt.viz and viz_path.exists():
            do_viz = False
        if opt.viz and opt.eval and viz_eval_path.exists():
            do_viz_eval = False
        timer.update("load_cache")

    # if not (do_match or do_eval or do_viz or do_viz_eval):
    #     timer.print("Finished pair {:5} of {:5}".format(i, len(pairs)))

    # If a rotation integer is provided (e.g. from EXIF data), use it:
    if len(pair) >= 5:
        rot0, rot1 = int(pair[2]), int(pair[3])
    else:
        rot0, rot1 = 0, 0

    # Load the image pair.
    image0, inp0, scales0 = read_image(
        image_dir / name0, device, opt.resize, rot0, opt.resize_float
    )
    image1, inp1, scales1 = read_image(
        image_dir / name1, device, opt.resize, rot1, opt.resize_float
    )
    if image0 is None or image1 is None:
        print(
            "Problem reading image pair: {} {}".format(
                image_dir / name0, image_dir / name1
            )
        )
        exit(1)
    timer.update("load_image")

    if do_match:
        # Perform the matching.
        pred = matching({"image0": inp0, "image1": inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        matches, conf = pred["matches0"], pred["matching_scores0"]
        timer.update("matcher")

        # Write the matches to disk.
        out_matches = {
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "matches": matches,
            "match_confidence": conf,
        }
        np.savez(str(matches_path), **out_matches)

    # Organize match data
    count = 0
    for m in range(len(matches)):  # Calculates number of valid matches
        if matches[m] != -1:
            count += 1
    keypoint_pairs = -1 * np.ones((count, 4))
    inc = 0
    for n in range(len(matches)):
        if matches[n] != -1:

            keypoint_pairs[inc][0] = kpts0[n][0]
            keypoint_pairs[inc][1] = kpts0[n][1]
            keypoint_pairs[inc][2] = kpts1[matches[n]][0]
            keypoint_pairs[inc][3] = kpts1[matches[n]][1]
            # Every element of keypoint pairs will be list of 4 values, the x,y coords of a point in img1, and then x,y of corresponding point in img2
            inc += 1
    print(f"Done")

    # load reconstruction from colmap
    cameras, images, points = read_model(path=opt.input_dir + "/sparse", ext=".bin")

    print(f"Cameras: {len(cameras)}")
    print(f"Images: {len(images)}")
    print(f"3D points: {len(points)}")

    indices = [c for c in cameras]
    dataList = [0, 1]
    threshold = 1.5

    for j in range(
        2
    ):  # range(2) is hard-coded here, it will only run get_image on 2 images, as we are only matching those 2 right now
        idx = indices[j]

        data = get_image(opt, str(input_dir), idx)
        dataList[
            j
        ] = data  # Each element in dataList is a dictionary containing the info about an image returned from get_image().
    data_one = {
        "R": dataList[0]["R"],
        "depth": dataList[0]["depth"],
        "T": dataList[0]["T"],
        "q": dataList[0]["q"],
        "K": dataList[0]["K"],
    }
    data_two = {
        "R": dataList[1]["R"],
        "depth": dataList[1]["depth"],
        "T": dataList[1]["T"],
        "q": dataList[1]["q"],
        "K": dataList[1]["K"],
    }
    validPoints1 = []
    validPoints2 = []
    groundTruthMatches = []
    gtCount = 0
    vpts0 = []
    vpts1 = []
    # For every pair of keypoint coordinates found by superglue:
    #  1) compare it to the coords of every point found in get_img/colmap (in dataList)
    #  2) Find the closest dataList point, save its location, distance to kpt, and colmap ID
    #  3) Repeat for keypoints found by superglue in second image:
    #  4) See if both keypoints are within threshold distance. If they are, AND point to the same point (colmap ID) add to g.t matches
    for j in range(len(keypoint_pairs)):
        sID1 = j
        sID2 = j
        x1s = keypoint_pairs[j][0]  # Superglue xys from image 1
        y1s = keypoint_pairs[j][1]
        x2s = keypoint_pairs[j][2]  # Superglue xys from image 2
        y2s = keypoint_pairs[j][3]
        distance1 = 10000000
        for k in range(len(dataList[0]["xys"])):
            x1c = dataList[0]["xys"][k][0]  # Colmap xys from image 1
            y1c = dataList[0]["xys"][k][1]
            tempDistance1 = findDistance(x1s, y1s, x1c, y1c)
            if tempDistance1 < distance1:
                distance1 = tempDistance1
                gtID1 = dataList[0]["ids"][
                    k
                ]  # this is the ground truth ID of the closest point to sg point

        distance2 = 1000000
        for k in range(len(dataList[1]["xys"])):
            x2c = dataList[1]["xys"][k][0]  # Colmap xys form image 2
            y2c = dataList[1]["xys"][k][1]
            tempDistance2 = findDistance(x2s, y2s, x2c, y2c)
            if tempDistance2 < distance2:
                distance2 = tempDistance2
                gtID2 = dataList[1]["ids"][
                    k
                ]  # this is the ground truth ID of the closest point to sg point

        if distance1 < threshold and distance2 < threshold:
            validPoints1.append(
                [sID1, x1s, y1s, distance1, gtID1, "img0"]
            )  # valid points1 will have every superglue keypoint from 1st image: ID,xylocation,dist to gtpt, gtptID
            validPoints2.append(
                [sID2, x2s, y2s, distance2, gtID2, "img1"]
            )  # validpoints2 has sg points from img 2
            gtCount += 1
            vpts0.append([x1s, y1s])
            vpts1.append([x2s, y2s])

    for f in range(gtCount):

        if (
            validPoints1[f][0] == validPoints2[f][0]
            and validPoints1[f][4] == validPoints2[f][4]
            and gtCount > 0
        ):

            print("ADDED TO GROUND TRUTH MATCHES")
            groundTruthMatches.append(
                [
                    validPoints1[f],
                    validPoints1[f][4],
                    validPoints1[f][3],
                    validPoints2[f][3],
                ]
            )
    # Next: get validPoints in a format to pass to the graphing funcs
    # Keep the matching keypoints. **Next do own version of this but the threshold isn't just if -1 determined by s.glue but by colmap comaprison
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
    # return pair, mkpts0, mkpts1, data_one, data_two
    normed_error = []
    for i in range(len(mkpts0)):  # len(mkpts0)
        # normed_error.append(
        #     calculate_distance(pair, mkpts0[i], mkpts1[i], data_one, data_two)
        # )
        normed_error.append(
            reprojection_error(pair, mkpts0[i], mkpts1[i], data_one, data_two)
        )
    if do_eval:
        # Estimate the pose and compute the pose error.
        assert len(pair) == 38, "Pair does not have ground truth info"
        K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
        K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
        T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

        # Scale the intrinsics to resized image.
        K0 = scale_intrinsics(K0, scales0)
        K1 = scale_intrinsics(K1, scales1)

        # Update the intrinsics + extrinsics if EXIF rotation was found.
        if rot0 != 0 or rot1 != 0:
            cam0_T_w = np.eye(4)
            cam1_T_w = T_0to1
            if rot0 != 0:
                K0 = rotate_intrinsics(K0, image0.shape, rot0)
                cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            if rot1 != 0:
                K1 = rotate_intrinsics(K1, image1.shape, rot1)
                cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
            cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
            T_0to1 = cam1_T_cam0

        epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
        correct = epi_errs < 5e-4
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else 0
        matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

        thresh = 1.0  # In pixels relative to resized image size.
        ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        # Write the evaluation results to disk.
        out_eval = {
            "error_t": err_t,
            "error_R": err_R,
            "precision": precision,
            "matching_score": matching_score,
            "num_correct": num_correct,
            "epipolar_errors": epi_errs,
        }
        np.savez(
            str(eval_path), **out_eval
        )  # np.savez to dump_match_pairs, but this is eval not coords
        timer.update("eval")

    if do_viz:
        # Visualize the matches.
        color = cm.jet(mconf)
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
            "Matches: {}".format(len(mkpts0)),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append("Rotation: {}:{}".format(rot0, rot1))

        # Display extra parameter info.
        k_thresh = matching.superpoint.config["keypoint_threshold"]
        m_thresh = matching.superglue.config["match_threshold"]
        small_text = [
            "Keypoint Threshold: {:.4f}".format(k_thresh),
            "Match Threshold: {:.2f}".format(m_thresh),
            "Image Pair: {}:{}".format(stem0, stem1),
        ]

        make_matching_plot(
            image0,
            image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            viz_path,
            opt.show_keypoints,
            opt.fast_viz,
            opt.opencv_display,
            "Matches",
            small_text,
        )

        # changed this section version to use vpts, but throws error.
        # vpts instead of kpts too
        # make_matching_plot(
        #     image0, image1, vpts0, vpts1, vpts0, vpts1, color,
        #     text, viz_path, opt.show_keypoints,
        #     opt.fast_viz, opt.opencv_display, 'Matches', small_text
        # )

        timer.update("viz_match")

    print("Normed errors:")
    print(normed_error)
    color = np.clip((np.array(normed_error) - 0) / (1e-3 - 0), 0, 1)
    color = error_colormap(1 - color)
    make_matching_plot(
        image0,
        image1,
        kpts0,
        kpts1,
        mkpts0[:10],
        mkpts1[:10],
        color,
        text,
        viz_eval_path,
        opt.show_keypoints,
        opt.fast_viz,
        opt.opencv_display,
        "Relative Pose",
        small_text,
    )

    if do_viz_eval:
        # Visualize the evaluation results for the image pair.
        # color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
        color = np.clip((normed_error - 0) / (1e-3 - 0), 0, 1)
        color = error_colormap(1 - color)
        deg, delta = " deg", "Delta "
        if not opt.fast_viz:
            deg, delta = "°", "$\\Delta$"
        e_t = "FAIL" if np.isinf(err_t) else "{:.1f}{}".format(err_t, deg)
        e_R = "FAIL" if np.isinf(err_R) else "{:.1f}{}".format(err_R, deg)
        text = [
            "SuperGlue",
            "{}R: {}".format(delta, e_R),
            "{}t: {}".format(delta, e_t),
            "inliers: {}/{}".format(num_correct, (matches > -1).sum()),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append("Rotation: {}:{}".format(rot0, rot1))

        # Display extra parameter info (only works with --fast_viz).
        k_thresh = matching.superpoint.config["keypoint_threshold"]
        m_thresh = matching.superglue.config["match_threshold"]
        small_text = [
            "Keypoint Threshold: {:.4f}".format(k_thresh),
            "Match Threshold: {:.2f}".format(m_thresh),
            "Image Pair: {}:{}".format(stem0, stem1),
        ]

        make_matching_plot(
            image0,
            image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            viz_eval_path,
            opt.show_keypoints,
            opt.fast_viz,
            opt.opencv_display,
            "Relative Pose",
            small_text,
        )

        timer.update("viz_eval")
    timer.print("Finished pairwise evaluation")

    # timer.print("Finished pair {:5} of {:5}".format(i, len(pairs)))
