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

    # ── Phase A: cached boolean visibility mask for sparse MLP evaluation ──
    # Initialized to None; set each iteration from the rasterizer's radii output.
    # After densification the Gaussian count changes, so we detect size mismatch
    # and fall back to evaluating the full set for that single step.
    prev_vis_mask: torch.Tensor | None = None

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
        # SPECULAR (SG STYLE) — Phase A: sparse MLP via cached visibility
        # --------------------------------------------------------
        # Only Gaussians visible in the PREVIOUS step feed the ASG MLP.
        # Typically 10–30 % of Gaussians are on-screen per frame, so this
        # cuts MLP forward+backward cost by 3–10×.
        # After densification the count changes → we detect size mismatch
        # and fall back to all Gaussians for that one step.

        spec_sparse: torch.Tensor | None = None  # sparse MLP output [M, 3]
        vis_indices: torch.Tensor | None = None   # indices of evaluated Gaussians
        mlp_color: torch.Tensor | None = None     # full-scene buffer  [N, 3]

        if iteration > 3000:
            n_gs = gaussians.get_xyz.shape[0]
            asg_feat = gaussians.get_asg_features  # [N, 24]

            # Determine which Gaussians to evaluate
            if prev_vis_mask is not None and prev_vis_mask.shape[0] == n_gs:
                vis_indices = prev_vis_mask.nonzero(as_tuple=False).squeeze(1)  # [M]
            else:
                # First specular step OR count changed after densification
                vis_indices = torch.arange(n_gs, device="cuda")

            if vis_indices.numel() > 0:
                spec_sparse = specular_mlp.step(
                    asg_feat[vis_indices],
                    viewdir[vis_indices],
                    normal[vis_indices],
                )  # [M, 3]

                # Scatter back into a full-scene buffer; index_put preserves grad
                mlp_color = torch.zeros(
                    (n_gs, 3), device="cuda"
                ).index_put((vis_indices,), spec_sparse)

        # --------------------------------------------------------
        # RENDER  (single pass — Phase A removes redundant sh & spec-sharp passes)
        # --------------------------------------------------------

        render_pkg = render_fastgs(
            cam,
            gaussians,
            pipe,
            background,
            opt.mult,
            mlp_color=mlp_color
        )

        image                 = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter     = render_pkg["visibility_filter"]
        radii                 = render_pkg["radii"]

        # ── Update cached visibility mask for the NEXT iteration ──────────
        # radii has shape [N]; (radii > 0) gives the boolean visibility mask.
        # We store it here so the next step uses it without an extra render.
        prev_vis_mask = (radii > 0)  # [N] bool

        # --------------------------------------------------------
        # LOSS
        # --------------------------------------------------------
        # Phase A design:
        #   • photometric_loss  — L1 + SSIM, unchanged from FastGS baseline.
        #     Gradients flow back through the renderer into mlp_color, so the
        #     specular MLP is already supervised by image reconstruction.
        #   • spec_reg          — lightweight L2 penalty on specular MLP outputs
        #     (Gaussian-space). Prevents the MLP from producing unbounded colors
        #     or collapsing to zero, without requiring a second render pass.
        # --------------------------------------------------------

        gt = cam.original_image.cuda()

        Ll1      = l1_loss(image, gt)
        ssim_val = fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))

        photometric_loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim_val)
        )

        # Gaussian-space specular regulariser (no extra render needed)
        if spec_sparse is not None:
            spec_reg = spec_sparse.pow(2).mean() * 0.01
        else:
            spec_reg = 0.0

        loss = photometric_loss + spec_reg

        loss.backward()

        # --------------------------------------------------------
        # Reduce SH competition: scale down SH gradients for a window after specular activation
        # This prevents SH from absorbing highlight residuals while specular learns.
        # Minimal parameters (tunable): spec_start, spec_freeze_steps, sh_grad_scale
        # --------------------------------------------------------
        spec_start = 3000  # must match when specular is enabled
        spec_freeze_steps = 2000  # number of iterations to reduce SH learning
        sh_grad_scale = 0.01  # scale applied to SH gradients (0 disables updates)

        if iteration > spec_start and iteration <= spec_start + spec_freeze_steps:
            try:
                if hasattr(gaussians, "_features_rest") and gaussians._features_rest.grad is not None:
                    gaussians._features_rest.grad.mul_(sh_grad_scale)
                if hasattr(gaussians, "_features_dc") and gaussians._features_dc.grad is not None:
                    gaussians._features_dc.grad.mul_(sh_grad_scale)
            except Exception:
                pass

        # --------------------------------------------------------
        # OPTIMIZER STEP
        # --------------------------------------------------------

        skip_sh = (iteration > spec_start and iteration <= spec_start + spec_freeze_steps)
        gaussians.optimizer_step(iteration, skip_sh=skip_sh)

        # Update specular lr BEFORE stepping optimizer to ensure non-zero lr is used
        specular_mlp.update_learning_rate(iteration)
        specular_mlp.optimizer_step()

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

                with torch.no_grad():
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

