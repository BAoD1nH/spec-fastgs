# ============================================================
# Rendering Script (Spec-Gaussian + FastGS)
# ============================================================

import torch
import os
import time
from tqdm import tqdm
from os import makedirs
import torchvision

from scene import Scene, GaussianModel, SpecularModel
from gaussian_renderer import render_fastgs

from utils.general_utils import safe_state

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args


# ------------------------------------------------------------
# RENDER ONE SET (TRAIN / TEST)
# ------------------------------------------------------------

def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    specular_mlp,
    args
):

    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    spec_path = os.path.join(model_path, name, f"ours_{iteration}", "spec")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(spec_path, exist_ok=True)

    total_time = 0.0

    for idx, view in enumerate(tqdm(views, desc=f"{name} rendering")):

        # --------------------------------------------------------
        # COMPUTE VIEWDIR + NORMAL
        # --------------------------------------------------------

        xyz = gaussians.get_xyz
        cam_center = view.camera_center.to("cuda")  

        viewdir = xyz - cam_center
        viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-6)

        normal = gaussians.get_normal_axis(viewdir).to("cuda")

        # --------------------------------------------------------
        # SPECULAR
        # --------------------------------------------------------

        mlp_color = specular_mlp.step(
            gaussians.get_asg_features.to("cuda"),
            viewdir,
            normal
        )

        # --------------------------------------------------------
        # DEBUG VISUALIZATIONS (SH-only, specular diagnostics)
        # --------------------------------------------------------
        # Minimal extra renders: compute SH-only image and full image

        # SH-only render (no specular)
        sh_pkg = render_fastgs(
            view,
            gaussians,
            pipeline,
            background,
            args.mult,
            mlp_color=None
        )
        sh_image = sh_pkg["render"]

        # Full render (SH + specular)
        start_time = time.time()

        render_pkg = render_fastgs(
            view,
            gaussians,
            pipeline,
            background,
            args.mult,
            mlp_color=mlp_color
        )

        end_time = time.time()
        total_time += (end_time - start_time)

        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]

        # Specular image = full - SH
        spec_image = rendering - sh_image

        # 1) Amplified specular (for visibility)
        amp_factor = 10.0
        spec_scaled = (spec_image * amp_factor).clamp(0.0, 1.0)
        torchvision.utils.save_image(
            spec_scaled,
            os.path.join(spec_path, f"{idx:05d}_spec_scaled.png")
        )

        # 2) Normalized specular (min-max to [0,1])
        minv = spec_image.min()
        maxv = spec_image.max()
        if (maxv - minv).abs() > 1e-8:
            spec_norm = (spec_image - minv) / (maxv - minv)
        else:
            spec_norm = spec_image.clone()
        spec_norm = spec_norm.clamp(0.0, 1.0)
        torchvision.utils.save_image(
            spec_norm,
            os.path.join(spec_path, f"{idx:05d}_spec_norm.png")
        )

        # 3) Residual (absolute GT - SH) and scaled for visualization
        residual = (gt - sh_image).abs()
        residual_scale = 5.0
        residual_vis = (residual * residual_scale).clamp(0.0, 1.0)
        torchvision.utils.save_image(
            residual_vis,
            os.path.join(spec_path, f"{idx:05d}_residual.png")
        )

        # --------------------------------------------------------
        # SAVE RENDERS (original behavior)
        # --------------------------------------------------------

        torchvision.utils.save_image(
            rendering,
            os.path.join(render_path, f"{idx:05d}.png")
        )

        torchvision.utils.save_image(
            gt,
            os.path.join(gts_path, f"{idx:05d}.png")
        )

        # --------------------------------------------------------
        # SPECULAR SHARP PASS (minimal change)
        # Render specular only with reduced spatial smoothing by using
        # a smaller scaling_modifier for the specular pass. Do not
        # change the diffuse SH rendering above.
        # --------------------------------------------------------

        # Only run this pass if we actually have a specular contribution
        if mlp_color is not None:
            # smaller scaling modifier for sharper splats (tune 0.3-0.5)
            spec_scaling_modifier = 0.35

            # override_color set to zero so renderer composes only mlp_color
            zeros_override = torch.zeros((gaussians.get_xyz.shape[0], 3), dtype=gaussians.get_xyz.dtype, device="cuda")

            spec_sharp_pkg = render_fastgs(
                view,
                gaussians,
                pipeline,
                background,
                args.mult,
                mlp_color=mlp_color,
                scaling_modifier=spec_scaling_modifier,
                override_color=zeros_override
            )

            spec_sharp = spec_sharp_pkg["render"]

            # Save specular-only sharp render and a composite with the original SH
            torchvision.utils.save_image(
                spec_sharp.clamp(0.0, 1.0),
                os.path.join(spec_path, f"{idx:05d}_spec_sharp.png")
            )

            # Composite: original SH (from sh_image) + sharp specular
            composite_sharp = (sh_image + spec_sharp).clamp(0.0, 1.0)
            torchvision.utils.save_image(
                composite_sharp,
                os.path.join(spec_path, f"{idx:05d}_sharp_composite.png")
            )

        # --------------------------------------------------------
        # end frame loop
        # --------------------------------------------------------

    num_frames = len(views)
    avg_time = total_time / num_frames if num_frames > 0 else 0.0
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    print(f"[{name}] {num_frames} frames | FPS: {fps:.2f}")


# ------------------------------------------------------------
# MAIN RENDER FUNCTION
# ------------------------------------------------------------

def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args
):

    with torch.no_grad():

        # --------------------------------------------------------
        # LOAD MODELS
        # --------------------------------------------------------

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # ✅ LOAD ASG FEATURE
        asg_path = os.path.join(
            dataset.model_path,
            f"point_cloud/iteration_{scene.loaded_iter}/asg.pt"
        )

        print("Loading ASG from:", asg_path)

        gaussians._features_asg = torch.load(asg_path).cuda()

        print("ASG loaded with shape:", gaussians._features_asg.shape)
        
        specular_mlp = SpecularModel(dataset.is_real, dataset.is_indoor)
        specular_mlp.load_weights(dataset.model_path, iteration=scene.loaded_iter)

        # --------------------------------------------------------
        # BACKGROUND
        # --------------------------------------------------------

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # --------------------------------------------------------
        # RENDER TRAIN / TEST
        # --------------------------------------------------------

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                specular_mlp,
                args
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                specular_mlp,
                args
            )


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":

    parser = ArgumentParser(description="Rendering Spec-FastGS")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mult", type=float, default=0.5)

    args = get_combined_args(parser)

    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args
    )

