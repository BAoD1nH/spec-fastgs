# gaussian_renderer/__init__.py (Spec-FastGS version)

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
    get_flag=None,
    metric_map=None,
    specular_model=None,   # <<< THÊM
    specular_mask=None,    # <<< THÊM
    debug_specular_mask=False
):
    """
    FastGS renderer with optional specular augmentation.
    """

    # ========== Screen-space points (FastGS original) ==========
    screenspace_points = torch.zeros(
        (pc.get_xyz.shape[0], 4),
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device="cuda"
    )

    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map is None:
        metric_map = torch.zeros(
            int(viewpoint_camera.image_height * viewpoint_camera.image_width),
            dtype=torch.int,
            device="cuda"
        )

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
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
        get_flag=get_flag,
        metric_map=metric_map
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    opacity = pc.get_opacity

    # ========== Covariance ==========
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # ========== SH COLOR (FastGS original) ==========
    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = means3D - viewpoint_camera.camera_center[None, :]
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    # ========== SPECULAR ADD-ON ==========
    if specular_model is not None:
        with torch.set_grad_enabled(specular_model.training):
            viewdir = means3D - viewpoint_camera.camera_center[None, :]
            viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-6)

            # ⚠️ NORMAL TẠM THỜI – giữ đơn giản & ổn định
            normal = compute_gaussian_normals(pc)  # [N, 3]

            # ⚠️ FEAT TẠM THỜI (SpecularModel có thể dùng mask sau)
            feat = torch.zeros(
                (means3D.shape[0], specular_model.feat_dim),
                dtype=means3D.dtype,
                device=means3D.device
            )

            specular_color = specular_model(feat, viewdir, normal)

            if specular_mask is not None:
                specular_color[~specular_mask] = 0.0

        alpha = 0.3 # Hệ số điều chỉnh cường độ phản xạ – có thể tinh chỉnh
        specular_color = alpha * specular_color 

        if colors_precomp is not None:
            colors_precomp = colors_precomp + specular_color
        else:
            dc = dc + specular_color.unsqueeze(1)
        
        if debug_specular_mask and specular_mask is not None:
            colors_precomp = torch.zeros_like(colors_precomp)
            colors_precomp[specular_mask] = torch.tensor([1.0, 0.0, 0.0], device=colors_precomp.device, dtype=colors_precomp.dtype)  # red highlight for specular Gaussians

    # ========== Rasterize ==========
    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        dc=dc if colors_precomp is None else None,
        shs=shs if colors_precomp is None else None,
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