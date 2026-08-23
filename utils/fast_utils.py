import torch
from PIL import ImageFilter
from gaussian_renderer import render_fastgs
from .loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
import torchvision.transforms as transforms
import random


def prior_to_cuda(prior, dtype=torch.float32):
    """Decode a compact CPU uint8 prior only for the camera in use."""
    if prior is None:
        return None
    result = prior.to(device="cuda", dtype=dtype, non_blocking=True)
    if prior.dtype == torch.uint8:
        result = result / 255.0
    return result


def sampling_cameras(my_viewpoint_stack, num_cams=10):
    ''' Randomly sample a given number of cameras from the viewpoint stack'''

    camlist = []
    num_cams = min(num_cams, len(my_viewpoint_stack))
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))
    
    return camlist

def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

    return l1_loss_norm

def compute_photometric_loss(viewpoint_cam, image):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss

def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value

def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY, iteration=None):
    """Compute multi-view consistency scores for Gaussians to guide densification.

    For each camera in `camlist` the function renders the scene and computes a
    photometric loss and a binary metric map of high-error pixels. It accumulates
    per-Gaussian counts of views that flagged the Gaussian and a weighted
    photometric score across views.

    Args:
        camlist (list): list of viewpoint camera objects to render from.
        gaussians: current Gaussian representation (model/state) used for rendering.
        pipe: rendering pipeline/context required by `render`.
        bg: background used for rendering.
        args: runtime config containing thresholds (e.g. `loss_thresh`).
        DENSIFY (bool): whether to compute and return the importance score
            used for densification. If False, only the pruning score is computed.

    Returns:
        importance_score (Tensor): per-Gaussian integer counts of how many views
            marked the Gaussian as high-error (floor-averaged across views).
            This output is only returned if `DENSIFY` is True.
        pruning_score (Tensor): normalized (0..1) per-Gaussian score used to
            prioritize densification (higher means worse reconstruction consistency).
    """

    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        render_image = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult)["render"]
        photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image)

        gt_image = my_viewpoint_cam.original_image.cuda()
        get_flag = True
        l1_loss_norm = get_loss(render_image, gt_image)
            
        weighted_error = l1_loss_norm

        # Ref-score guides FastGS ADC; it does not spawn Gaussians by itself.
        use_ref_score = False
        if (getattr(args, 'use_ref_score', False) and hasattr(my_viewpoint_cam, 'ref_score')
                and iteration is not None and not getattr(args, 'disable_ref_score', False)):
            if iteration % args.densification_refscore_interval == 0:
                n_budget = getattr(args, 'max_refscore_gaussians', 0)
                n_current = gaussians.get_xyz.shape[0]
                if n_budget > 0 and n_current < n_budget:
                    ratio = min(max(n_current / n_budget, 0.0), 1.0)
                    decay_power = getattr(args, 'refscore_decay_power', 1.0)
                    min_strength = getattr(args, 'refscore_min_strength', 0.15)
                    strength = max((1.0 - ratio) ** decay_power, min_strength)
                    use_ref_score = True

        if use_ref_score:
            ref_score = prior_to_cuda(my_viewpoint_cam.ref_score, l1_loss_norm.dtype)
            ref_weight = float(getattr(args, 'refscore_strength', 0.5)) * strength
            weighted_error = l1_loss_norm * (1.0 + ref_weight * ref_score)

        metric_map = (weighted_error > args.loss_thresh).int()

        render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult, get_flag = get_flag, metric_map = metric_map)

        accum_loss_counts = render_pkg["accum_metric_counts"]

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone()
        else:
            full_metric_score += photometric_loss * accum_loss_counts

    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (torch.max(full_metric_score) - torch.min(full_metric_score))
    
    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None
    return importance_score, pruning_score
