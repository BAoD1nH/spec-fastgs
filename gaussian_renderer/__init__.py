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
    get_flag=False,                 # ✅ BẮT BUỘC cho pixel→Gaussian
    metric_map=None,                # ✅ internal buffer
    specular_model=None,
    specular_mask=None,
    debug_specular_mask=False
):
    """
    FastGS renderer with optional Specular (ASG) augmentation.
    Extended to expose pixel -> Gaussian mapping for specular detection.
    """

    # ------------------------------------------------------------
    # Screen-space placeholder (FastGS original)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Camera parameters
    # ------------------------------------------------------------
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)

    # ------------------------------------------------------------
    # ✅ Pixel → Gaussian ID buffer (BẮT BUỘC cho Cách A)
    # ------------------------------------------------------------
    if metric_map is None:
        metric_map = torch.zeros(
            H * W,
            dtype=torch.int32,
            device="cuda"
        )

    # ------------------------------------------------------------
    # Rasterizer settings
    # ------------------------------------------------------------
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
        get_flag=get_flag,           # ✅ BẮT BUỘC
        metric_map=metric_map        # ✅ BẮT BUỘC
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ------------------------------------------------------------
    # Gaussian parameters
    # ------------------------------------------------------------
    means3D = pc.get_xyz
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # ------------------------------------------------------------
    # SH color (FastGS original)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # ✅ SPECULAR ADD-ON (ASG)
    # ------------------------------------------------------------
    if specular_model is not None:
        with torch.set_grad_enabled(specular_model.training):
            viewdir = means3D - viewpoint_camera.camera_center[None, :]
            viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-6)

            # ✅ normal hình học đúng
            normal = compute_gaussian_normals(pc)

            # ✅ feature placeholder (Phase 3 có thể mở rộng)
            feat = torch.zeros(
                (means3D.shape[0], specular_model.feat_dim),
                dtype=means3D.dtype,
                device=means3D.device
            )

            specular_color = specular_model(feat, viewdir, normal)

            # ✅ Chỉ bật ASG tại Gaussian được detect
            if specular_mask is not None:
                specular_color[~specular_mask] = 0.0

        # ✅ strength control
        alpha = 0.3
        specular_color = alpha * specular_color

        if colors_precomp is not None:
            colors_precomp = colors_precomp + specular_color
        else:
            dc = dc + specular_color.unsqueeze(1)

        # --------------------------------------------------------
        # ✅ DEBUG: visualize specular Gaussian mask
        # --------------------------------------------------------
        if debug_specular_mask and specular_mask is not None:
            colors_precomp = torch.zeros_like(colors_precomp)
            colors_precomp[specular_mask] = torch.tensor(
                [1.0, 0.0, 0.0],
                device=colors_precomp.device,
                dtype=colors_precomp.dtype
            )

    # ------------------------------------------------------------
    # Rasterize
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # ✅ OUTPUT (BẮT BUỘC cho Cách A)
    # ------------------------------------------------------------
    return {
        "render": rendered_image,
        "gaussian_index_map": metric_map.view(H, W),  # ✅ PIXEL → GAUSSIAN
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "accum_metric_counts": accum_metric_counts
    }