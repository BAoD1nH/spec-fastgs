# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fastgs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    args,
    specular_model=None   # <<< THÊM
):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")

    total_time = 0.0

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_time = time.time()

        render_pkg = render_fastgs(
            view,
            gaussians,
            pipeline,
            background,
            args.mult,
            specular_model=specular_model   # <<< TRUYỀN XUỐNG
        )

        rendering = render_pkg["render"]

        end_time = time.time()
        total_time += (end_time - start_time)

        gt = view.original_image[0:3, :, :]

        torchvision.utils.save_image(
            rendering,
            os.path.join(render_path, f"{idx:05d}.png")
        )
        torchvision.utils.save_image(
            gt,
            os.path.join(gts_path, f"{idx:05d}.png")
        )

    num_frames = len(views)
    avg_time = total_time / num_frames if num_frames > 0 else 0.0
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    print(
        f"[{name}] Rendered {num_frames} frames in {total_time:.2f}s. "
        f"Average FPS: {fps:.2f}"
    )


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args
):
    with torch.no_grad():
        # === Load Gaussian Model (FastGS) ===
        gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # === Background ===
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # === OPTIONAL: Load specular model ===
        specular_model = None
        if args.use_specular:
            from specular import SpecularModel

            specular_model = SpecularModel().cuda()
            specular_ckpt = os.path.join(
                dataset.model_path, "specular", "specular.pth"
            )

            if not os.path.exists(specular_ckpt):
                raise FileNotFoundError(
                    f"[Spec-FastGS] Specular checkpoint not found: {specular_ckpt}"
                )

            specular_model.load_state_dict(torch.load(specular_ckpt))
            specular_model.eval()

            print("[Spec-FastGS] Specular model loaded.")

        # === Render train / test sets ===
        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                args,
                specular_model
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
                args,
                specular_model
            )


if __name__ == "__main__":

    parser = ArgumentParser(description="Rendering script (FastGS / Spec-FastGS)")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mult", type=float, default=0.5)

    # <<< THÊM FLAG SPECULAR >>>
    parser.add_argument(
        "--use_specular",
        action="store_true",
        help="Enable Specular-FastGS rendering"
    )


    parser.add_argument(
        "--debug_specular_mask",
        action="store_true",
        help="Visualize specular Gaussians"
    )

    args = get_combined_args(parser)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args
    )