# gaussian_renderer/__init__.py
# ============================================================
# Specular-aware FastGS Renderer
# Joint Diffuse (SH) + Specular (ASG), no post-hoc
# ============================================================

import torch
import math

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization_fastgs import (
    GaussianRasterizationSettings,
    GaussianRasterizer
)
from specular.normal_utils import compute_gaussian_normals


def render_fastgs(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    mult,
    scaling_modifier=1.0,
    override_color=None,
    specular_model=None,
):
    """
    Specular-aware FastGS renderer.

    final_color = diffuse_SH + specular_ASG

    - Diffuse: SH only (view-dependent but smooth)
    - Specular: ASG (sharp, physics-like)
    """

    # =========================================================
    # Screen-space placeholder (FastGS original)
    # =========================================================
    screenspace_points = torch.zeros(
        (pc.get_xyz.shape[0], 4),
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device="cuda"
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # =========================================================
    # Camera parameters
    # =========================================================
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)

    # =========================================================
    # Rasterizer setup (UNCHANGED from FastGS)
    # =========================================================
    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult=mult,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    opacity = pc.get_opacity

    # =========================================================
    # Covariance (FastGS original)
    # =========================================================
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # =========================================================
    # ====== DIFFUSE COMPONENT (SH ONLY, NO SPECULAR) ======
    # =========================================================
    shs_view = pc.get_features().transpose(1, 2).view(
        -1, 3, (pc.max_sh_degree + 1) ** 2
    )

    # view direction (world space)
    dir_pp = means3D - viewpoint_camera.camera_center[None, :]
    viewdir = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-6)

    # SH diffuse color
    rgb_diffuse = eval_sh(
        pc.active_sh_degree,
        shs_view,
        viewdir
    )
    rgb_diffuse = torch.clamp_min(rgb_diffuse + 0.5, 0.0)

    # =========================================================
    # ====== SPECULAR COMPONENT (ASG – FIRST CLASS) ======
    # =========================================================
    if specular_model is not None:
        # Geometry-derived normals
        normal = compute_gaussian_normals(pc)

        # Per-Gaussian latent specular code
        spec_feat = pc.get_specular_feat()

        # ASG predicts true specular RGB (NOT residual hack)
        rgb_specular = specular_model(
            spec_feat,
            viewdir,
            normal
        )
    else:
        rgb_specular = 0.0

    # =========================================================
    # Final color composition (NO ALPHA HACK)
    # =========================================================
    colors_precomp = rgb_diffuse + rgb_specular

    # =========================================================
    # Rasterization
    # =========================================================
    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        dc=None,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "accum_metric_counts": accum_metric_counts
    }