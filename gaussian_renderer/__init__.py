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
        get_flag=False,
        metric_map=None,
        specular_model=None,
        specular_mask=None,
        debug_specular_mask=False
    ):

    screenspace_points = torch.zeros(
        (pc.get_xyz.shape[0], 4),
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device="cuda"
    )
    screenspace_points.retain_grad()


    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)


    # ✅ FIX QUAN TRỌNG
    if metric_map is None:
        metric_map = torch.zeros((H, W), dtype=torch.int32, device="cuda")


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
        get_flag=get_flag,
        metric_map=metric_map
    )


    rasterizer = GaussianRasterizer(raster_settings=raster_settings)


    means3D = pc.get_xyz
    opacity = pc.get_opacity


    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        scales = None
        rotations = None
    else:
        cov3D_precomp = None
        scales = pc.get_scaling
        rotations = pc.get_rotation


    # SH
    shs = None
    colors_precomp = None


    if override_color is None:
        shs_view = pc.get_features.transpose(1, 2).view(
            -1, 3, (pc.max_sh_degree + 1) ** 2
        )


        viewdir = means3D - viewpoint_camera.camera_center[None, :]
        viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-6)

        viewdir_for_sh = viewdir.detach()   # BLock gradient

        colors_precomp = torch.clamp_min(
            eval_sh(pc.active_sh_degree, shs_view, viewdir_for_sh) + 0.5,
            0.0
        )
    else:
        colors_precomp = override_color


    # ✅ SPECULAR
    if specular_model is not None:
        viewdir = means3D - viewpoint_camera.camera_center[None, :]
        viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-6)


        normal = compute_gaussian_normals(pc)
        feat = pc.get_features_specular


        specular_color = specular_model(feat, viewdir, normal)


        if specular_mask is not None:
            specular_color[~specular_mask] = 0.0


        colors_precomp = colors_precomp + 1 * specular_color


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
        "gaussian_index_map": metric_map,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "accum_metric_counts": accum_metric_counts,
        "specular": specular_color if specular_model is not None else None,
    }