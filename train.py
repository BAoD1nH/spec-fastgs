# ============================================================
# Training (Spec-Gaussian + FastGS)
# ============================================================

import torch
import numpy as np
import os, time, sys
from random import randint
from tqdm import tqdm
import uuid

from fused_ssim import fused_ssim as fast_ssim
from utils.loss_utils import l1_loss
from utils.image_utils import psnr

from gaussian_renderer import render_fastgs
from scene import Scene, GaussianModel, SpecularModel

from utils.general_utils import safe_state
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except:
    TENSORBOARD_FOUND = False


# ============================================================
# TRAINING LOOP
# ============================================================

def training(dataset, opt, pipe):

    tb_writer = prepare_output_and_logger(dataset)

    # ------------------------------------------------------------
    # INIT MODELS
    # ------------------------------------------------------------

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    specular_mlp = SpecularModel(dataset.is_real, dataset.is_indoor)
    specular_mlp.train_setting(opt)

    # ------------------------------------------------------------
    # BG
    # ------------------------------------------------------------

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Training")
    ema_loss = 0.0

    for iteration in progress_bar:

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # --------------------------------------------------------
        # SAMPLE CAMERA
        # --------------------------------------------------------

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        idx = randint(0, len(viewpoint_indices) - 1)
        cam = viewpoint_stack.pop(idx)
        viewpoint_indices.pop(idx)

        # --------------------------------------------------------
        # COMPUTE VIEWDIR + NORMAL
        # --------------------------------------------------------

        xyz = gaussians.get_xyz
        cam_center = cam.camera_center

        viewdir = xyz - cam_center
        viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-6)

        normal = gaussians.get_normal_axis(viewdir)

        # --------------------------------------------------------
        # SPECULAR (SG STYLE)
        # --------------------------------------------------------

        if iteration > 3000:
            mlp_color = specular_mlp.step(
                gaussians.get_asg_features,
                viewdir,
                normal
            )
        else:
            mlp_color = None

        # --------------------------------------------------------
        # RENDER
        # --------------------------------------------------------

        # Minimal intervention: compute SH-only render to get the residual
        # and supervise the specular contribution in image space.
        # Note: this does an extra (SH-only) render per step for testing.

        # SH-only render (no specular)
        sh_pkg = render_fastgs(
            cam,
            gaussians,
            pipe,
            background,
            opt.mult,
            mlp_color=None
        )
        sh_image = sh_pkg["render"]

        # Full render (SH + spec if mlp_color is not None)
        render_pkg = render_fastgs(
            cam,
            gaussians,
            pipe,
            background,
            opt.mult,
            mlp_color=mlp_color
        )

        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # --------------------------------------------------------
        # LOSS (NO SPEC LOSS) -> now augmented with spec_loss
        # --------------------------------------------------------

        gt = cam.original_image.cuda()

        # Photometric loss (original)
        Ll1 = l1_loss(image, gt)
        ssim_val = fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))

        photometric_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

        # Image-space spec supervision (residual)
        # residual := ground-truth - SH_only
        residual = gt - sh_image
        spec_image = image - sh_image

        # Mask residual to focus on meaningful pixels (optional)
        residual_threshold = 0.03  # tunable: pixels with mean-channel residual > thresh are used
        residual_mag = residual.abs().mean(dim=0)  # (H, W)
        mask = residual_mag > residual_threshold

        if mask.sum() > 0:
            mask3 = mask.unsqueeze(0).repeat(3, 1, 1)
            spec_loss = l1_loss(spec_image * mask3, residual * mask3)
        else:
            spec_loss = l1_loss(spec_image, residual)

        # Combine losses
        spec_weight = 1.0  # tunable weight for specular supervision
        loss = photometric_loss + spec_weight * spec_loss

        loss.backward()

        # --------------------------------------------------------
        # OPTIMIZER STEP
        # --------------------------------------------------------

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        specular_mlp.optimizer_step()
        specular_mlp.update_learning_rate(iteration)

        # --------------------------------------------------------
        # LOG
        # --------------------------------------------------------

        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss

        if iteration % 10 == 0:
            progress_bar.set_postfix({"loss": f"{ema_loss:.6f}"})

        # --------------------------------------------------------
        # DENSIFY (FASTGS)
        # --------------------------------------------------------

        if iteration < opt.densify_until_iter:

            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter]
            )

            gaussians.add_densification_stats(
                viewspace_point_tensor,
                visibility_filter
            )

            if (
                iteration > opt.densify_from_iter and
                iteration % opt.densification_interval == 0
            ):
                camlist = sampling_cameras(scene.getTrainCameras().copy())

                importance_score, pruning_score = compute_gaussian_score_fastgs(
                    camlist, gaussians, pipe, background, opt, DENSIFY=True
                )

                gaussians.densify_and_prune_fastgs(
                    max_screen_size=None,
                    min_opacity=0.005,
                    extent=scene.cameras_extent,
                    radii=radii,
                    args=opt,
                    importance_score=importance_score,
                    pruning_score=pruning_score
                )

        # --------------------------------------------------------
        # SAVE
        # --------------------------------------------------------

        if iteration in [17000, opt.iterations]:
            print(f"[ITER {iteration}] Saving...")
            scene.save(iteration)

            # ✅ QUAN TRỌNG NHẤT
            specular_mlp.save_weights(scene.model_path, iteration)

# ============================================================
# UTILS
# ============================================================

def prepare_output_and_logger(args):

    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)

    print("Output folder:", args.model_path)
    os.makedirs(args.model_path, exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    return None


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = ArgumentParser()

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args()

    safe_state(False)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args)
    )

    print("Training complete.")

