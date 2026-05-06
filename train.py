#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import torch
import numpy as np
import os, random, time, sys
from random import randint
from tqdm import tqdm
import uuid

from lpipsPyTorch import lpips
from fused_ssim import fused_ssim as fast_ssim

from gaussian_renderer import render_fastgs, network_gui_ws
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from utils.loss_utils import l1_loss
from utils.image_utils import psnr

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ============================================================
# TRAINING LOOP (SPECULAR-AWARE FASTGS)
# ============================================================
def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    websockets
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # === Gaussian Model (Specular-aware) ===
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    ema_loss_for_log = 0.0
    total_time = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # ========================================================
    # MAIN TRAINING LOOP
    # ========================================================
    for iteration in range(first_iter, opt.iterations + 1):

        # Web GUI (unchanged)
        if websockets and 0 <= network_gui_ws.curr_id < len(scene.getTrainCameras()):
            cam = scene.getTrainCameras()[network_gui_ws.curr_id]
            net_image = render_fastgs(
                cam, gaussians, pipe, background, opt.mult
            )["render"]
            network_gui_ws.latest_width = cam.image_width
            network_gui_ws.latest_height = cam.image_height
            network_gui_ws.latest_result = memoryview(
                (torch.clamp(net_image, 0, 1) * 255)
                .byte().permute(1, 2, 0).contiguous().cpu().numpy()
            )

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Increase SH degree gradually (FastGS original)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        # === RENDER (SPECULAR-AWARE) ===
        render_pkg = render_fastgs(
            viewpoint_cam, gaussians, pipe, bg, opt.mult
        )

        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # === LOSS ===
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Logging
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)

            iter_time = iter_start.elapsed_time(iter_end)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # ====================================================
            # DENSIFICATION / PRUNING (FASTGS ORIGINAL)
            # ====================================================
            optim_start.record()

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    camlist = sampling_cameras(scene.getTrainCameras().copy())
                    importance_score, pruning_score = compute_gaussian_score_fastgs(
                        camlist, gaussians, pipe, bg, opt, DENSIFY=True
                    )
                    gaussians.densify_and_prune_fastgs(
                        max_screen_size=size_threshold,
                        min_opacity=0.005,
                        extent=scene.cameras_extent,
                        radii=radii,
                        args=opt,
                        importance_score=importance_score,
                        pruning_score=pruning_score
                    )

                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Aggressive pruning stage
            if iteration % 3000 == 0 and 15000 < iteration < 30000:
                camlist = sampling_cameras(scene.getTrainCameras().copy())
                _, pruning_score = compute_gaussian_score_fastgs(
                    camlist, gaussians, pipe, bg, opt
                )
                gaussians.final_prune_fastgs(
                    min_opacity=0.1, pruning_score=pruning_score
                )

            # Optimizer step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)

            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    print(f"Gaussian number: {gaussians.get_xyz.shape[0]}")
    print(f"Training time: {total_time:.2f}s")


# ============================================================
# UTILS
# ============================================================
def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID", str(uuid.uuid4()))
        args.model_path = os.path.join("./output/", unique_str)

    print("Output folder:", args.model_path)
    os.makedirs(args.model_path, exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available")
        return None


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--websockets", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    safe_state(args.quiet)

    if args.websockets:
        network_gui_ws.init(args.ip, args.port)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.websockets
    )

    print("\nTraining complete.")
