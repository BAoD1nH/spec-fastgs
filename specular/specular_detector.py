# fastgs/specular/specular_detector.py

import torch
import numpy as np
from gaussian_renderer import render_fastgs
from utils.fast_utils import sampling_cameras
from .specular_config import *
from utils.loss_utils import l1_loss


@torch.no_grad()
def detect_specular_gaussians(scene, gaussians, pipeline, background, mult):
    """
    Detect Gaussians with VIEW-DEPENDENT STRUCTURED residual (specular/reflective).

    Returns:
        specular_mask: BoolTensor [N]
    """

    device = gaussians.get_xyz.device
    N = gaussians.get_xyz.shape[0]

    # Collect per-Gaussian error across views
    per_view_errors = []

    cameras = sampling_cameras(scene.getTrainCameras().copy())
    cameras = cameras[:NUM_SAMPLE_VIEWS]

    for cam in cameras:
        pkg = render_fastgs(
            cam,
            gaussians,
            pipeline,
            background,
            mult
        )

        render_img = pkg["render"]                  # [3,H,W]
        gt_img = cam.original_image[:3].to(device) # [3,H,W]
        visibility = pkg["visibility_filter"].squeeze()

        # Pixel L1 error (mean over RGB)
        pixel_err = torch.mean(torch.abs(render_img - gt_img), dim=0)  # [H,W]

        # Binary mask of high-error pixels
        high_err_mask = (pixel_err > PIXEL_ERROR_THRESH)

        # For each Gaussian: does it contribute to any high-error pixel?
        g_err = torch.zeros((N,), device=device)
        if visibility.numel() > 0:
            flat_err = pixel_err.view(-1)  # [H*W]

            k = max(1, int(0.001 * flat_err.numel()))  # top 0.1% pixels
            topk_err = torch.topk(flat_err, k=k, largest=True).values
            
            local_view_score = topk_err.mean() / max(len(visibility), 1)  # Normalize by visibility count

            g_err[visibility] = local_view_score

        per_view_errors.append(g_err)

    # [num_views, N] -> [N, num_views]
    E = torch.stack(per_view_errors, dim=1)

    # ---- Statistics per Gaussian ----
    mean_e = torch.mean(E, dim=1)
    var_e = torch.var(E, dim=1)
    max_e = torch.max(E, dim=1).values

    peak_ratio = max_e / (mean_e + 1e-6)
    active_views = (E > 0).sum(dim=1)

    # ---- Core specular conditions ----
    specular_candidate = (
        (active_views >= MIN_VIEWS_FOR_STATS) &
        (var_e >= VARIANCE_THRESH) &
        (peak_ratio >= PEAK_RATIO_THRESH)
    )

    # ---- Enforce sparsity (TOP-K) ----
    candidate_idx = torch.where(specular_candidate)[0]
    max_keep = int(MAX_SPECULAR_RATIO * N)

    if candidate_idx.numel() > max_keep:
        scores = var_e[candidate_idx] * peak_ratio[candidate_idx]
        topk = torch.topk(scores, k=max_keep).indices
        final_idx = candidate_idx[topk]
    else:
        final_idx = candidate_idx

    specular_mask = torch.zeros((N,), dtype=torch.bool, device=device)
    specular_mask[final_idx] = True

    if PRINT_STATS and specular_mask.any():
        print(
            f"[Specular-Detect] {specular_mask.float().mean()*100:.2f}% "
            f"Gaussians marked as specular | "
            f"mean(VAR)={var_e[specular_mask].mean():.6f} | "
            f"max(VAR)={var_e.max():.6f}"
        )
    elif PRINT_STATS:
        print("[Specular-Detect] 0.00% Gaussians marked as specular")

    return specular_mask