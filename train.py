# ============================================================
# Training (Spec-Gaussian + FastGS)
# ============================================================

import torch
import numpy as np
import os, time, sys, json, math
from io import BytesIO
from random import randint
from tqdm import tqdm
import uuid

from fused_ssim import fused_ssim as fast_ssim
from utils.loss_utils import l1_loss
from utils.image_utils import psnr

from gaussian_renderer import render_fastgs
from scene import Scene, GaussianModel, SpecularModel
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix

from utils.general_utils import safe_state
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras, prior_to_cuda
from utils.gaussian_heatmap import save_gaussian_view_heatmaps

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except:
    TENSORBOARD_FOUND = False


def configure_refscore_budget(opt, initial_gaussians):
    if not getattr(opt, 'use_ref_score', False):
        return

    if getattr(opt, 'max_refscore_gaussians', 400000) == -1:
        budget = int(initial_gaussians * opt.refscore_budget_multiplier)
        budget = max(budget, opt.refscore_budget_min)
        budget = min(budget, opt.refscore_budget_max)
        opt.max_refscore_gaussians = budget
        print(
            f"[Auto Budget] Ref Score cap: {budget:,} Gaussians "
            f"(initial={initial_gaussians:,}, multiplier={opt.refscore_budget_multiplier})"
        )
    else:
        print(f"[Ref Score Budget] cap: {opt.max_refscore_gaussians:,} Gaussians")


def _tensor_to_jpeg(image, quality=86):
    """Encode a CUDA CHW float image without requiring extra web packages."""
    from PIL import Image
    pixels = (image.detach().clamp(0, 1).mul(255).byte()
              .permute(1, 2, 0).contiguous().cpu().numpy())
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="JPEG", quality=quality)
    return output.getvalue()


def _make_orbit_camera(base_camera, target, settings, width, height):
    """Create a browser-controlled MiniCam around a fixed scene target."""
    base_eye = base_camera.camera_center.detach().cpu().numpy()
    target = np.asarray(target, dtype=np.float32)
    offset = base_eye - target
    radius = max(float(np.linalg.norm(offset)), 1e-3) * float(settings["zoom"])
    base_yaw = math.atan2(float(offset[0]), float(offset[2]))
    base_pitch = math.asin(float(np.clip(offset[1] / max(np.linalg.norm(offset), 1e-6), -1, 1)))
    yaw = base_yaw + float(settings["yaw"])
    pitch = np.clip(base_pitch + float(settings["pitch"]), -1.45, 1.45)
    eye = target + radius * np.array([
        math.cos(pitch) * math.sin(yaw), math.sin(pitch),
        math.cos(pitch) * math.cos(yaw)
    ], dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8
    world_up = np.array([0, 1, 0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-5:
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-8
    down = np.cross(forward, right)
    roll = float(settings.get("roll", 0.0))
    if roll:
        cos_roll, sin_roll = math.cos(roll), math.sin(roll)
        original_right, original_down = right.copy(), down.copy()
        right = cos_roll * original_right + sin_roll * original_down
        down = -sin_roll * original_right + cos_roll * original_down
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ eye
    w2v = np.eye(4, dtype=np.float32)
    w2v[:3, :3] = rotation
    w2v[:3, 3] = translation
    world_view = torch.tensor(w2v).transpose(0, 1).cuda()

    fov_scale = float(settings["fov_scale"])
    fovx = min(float(base_camera.FoVx) * fov_scale, math.radians(150))
    fovy = min(2 * math.atan(math.tan(fovx / 2) * height / width), math.radians(150))
    projection = getProjectionMatrix(
        znear=base_camera.znear, zfar=base_camera.zfar, fovX=fovx, fovY=fovy
    ).transpose(0, 1).cuda()
    return MiniCam(width, height, fovy, fovx, base_camera.znear,
                   base_camera.zfar, world_view, world_view @ projection)


def _geometry_colors(gaussians):
    """High-contrast scale encoding that exposes individual Gaussian splats."""
    scale = gaussians.get_scaling.detach().mean(dim=1)
    lo, hi = torch.quantile(scale, 0.05), torch.quantile(scale, 0.95)
    value = ((scale - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    return torch.stack((0.18 + 0.82 * value, 0.95 - 0.62 * value,
                        0.95 * (1.0 - value)), dim=1)


def _render_web_pair(base_camera, target, settings, width, height, gaussians,
                     specular_mlp, pipe, background, mult, use_asg,
                     capture_all=False):
    """Render synchronized geometry/RGB frames for live and final-viewer modes."""
    component = str(settings.get("rgb_component", "render"))
    # A residual is only meaningful when prediction and GT use the exact same
    # calibrated camera. Orbit controls intentionally do not affect this mode.
    orbit_cam = _make_orbit_camera(base_camera, target, settings, width, height)
    viewer_cam = base_camera if component == "residual_remaining" else orbit_cam

    def full_render(camera):
        viewer_xyz = gaussians.get_xyz
        viewer_dir = viewer_xyz - camera.camera_center
        viewer_dir = viewer_dir / (viewer_dir.norm(dim=1, keepdim=True) + 1e-6)
        viewer_normal = gaussians.get_normal_axis(viewer_dir)
        viewer_spec = None
        if use_asg:
            viewer_spec = specular_mlp.step(
                gaussians.get_asg_features, viewer_dir, viewer_normal
            )
        return render_fastgs(
            camera, gaussians, pipe, background, mult, mlp_color=viewer_spec
        )["render"]

    full_frame = full_render(viewer_cam)
    sh_frame = None
    if component in {"sh_only", "asg_only"} or capture_all:
        sh_frame = render_fastgs(
            viewer_cam, gaussians, pipe, background, mult, mlp_color=None
        )["render"]
    if component == "sh_only":
        rgb_frame = sh_frame
    elif component == "asg_only":
        rgb_frame = (full_frame - sh_frame).clamp_min(0.0)
    elif component == "residual_remaining":
        rgb_frame = (base_camera.original_image[:3].cuda() - full_frame).clamp_min(0.0)
    else:
        rgb_frame = full_frame
    geometry_opacity = torch.full_like(
        gaussians.get_opacity, float(settings["geometry_opacity"])
    )
    geometry_frame = render_fastgs(
        orbit_cam, gaussians, pipe,
        torch.tensor([0.015, 0.02, 0.022], device="cuda"), mult,
        scaling_modifier=float(settings["splat_scale"]),
        override_color=_geometry_colors(gaussians),
        opacity_override=geometry_opacity,
    )["render"]
    rgb_jpeg = _tensor_to_jpeg(rgb_frame)
    geometry_jpeg = _tensor_to_jpeg(geometry_frame)
    captures = None
    if capture_all:
        # Timeline RGB/SH/ASG use the orbit pose captured at this iteration.
        # Residual uses its calibrated dataset pose so prediction and GT align.
        if viewer_cam is not orbit_cam:
            orbit_full = full_render(orbit_cam)
            orbit_sh = render_fastgs(
                orbit_cam, gaussians, pipe, background, mult, mlp_color=None
            )["render"]
        else:
            orbit_full, orbit_sh = full_frame, sh_frame
        exact_full = full_frame if viewer_cam is base_camera else full_render(base_camera)
        residual = (base_camera.original_image[:3].cuda() - exact_full).clamp_min(0.0)
        captures = {
            "render": _tensor_to_jpeg(orbit_full),
            "sh_only": _tensor_to_jpeg(orbit_sh),
            "asg_only": _tensor_to_jpeg((orbit_full - orbit_sh).clamp_min(0.0)),
            "residual_remaining": _tensor_to_jpeg(residual),
            "geometry": geometry_jpeg,
        }
    return rgb_jpeg, geometry_jpeg, captures


def _write_web_live(live_dir, iteration, gaussian_count, allocated_mib,
                    reserved_mib, rgb_jpeg, geometry_jpeg, phase="training"):
    for name, encoded in (("rgb", rgb_jpeg), ("geometry", geometry_jpeg)):
        live_path = os.path.join(live_dir, name + ".jpg")
        temporary_path = live_path + ".tmp"
        with open(temporary_path, "wb") as frame_file:
            frame_file.write(encoded)
        os.replace(temporary_path, live_path)
    telemetry_path = os.path.join(live_dir, "telemetry.json")
    telemetry_tmp = telemetry_path + ".tmp"
    with open(telemetry_tmp, "w") as telemetry_file:
        json.dump({
            "type": "frame", "iteration": iteration,
            "gaussian_count": gaussian_count,
            "vram_allocated_mib": round(allocated_mib, 2),
            "vram_reserved_mib": round(reserved_mib, 2),
            "phase": phase, "frame_id": time.time_ns(),
        }, telemetry_file)
    os.replace(telemetry_tmp, telemetry_path)


def _merged_web_settings(web_viewer, settings_path):
    settings = web_viewer.poll_settings()
    if os.path.isfile(settings_path):
        try:
            with open(settings_path, "r") as settings_file:
                settings.update(json.load(settings_file))
        except (OSError, json.JSONDecodeError):
            pass
    return settings
def build_ref_score_confidence(ref_score, opt):
    """
    Keep broad RefScore for densification, but derive a conservative confidence
    map for supervision/masking where false positives are more damaging.
    """
    conf = prior_to_cuda(ref_score).detach().float().clamp(0.0, 1.0)
    if conf.numel() == 0 or conf.max() <= 0:
        return conf

    quantile = float(getattr(opt, "refscore_conf_quantile", 0.0))
    if 0.0 < quantile < 1.0:
        pivot = torch.quantile(conf.reshape(-1), quantile)
        conf_max = conf.max()
        if conf_max > pivot + 1e-6:
            conf = ((conf - pivot) / (conf_max - pivot + 1e-6)).clamp(0.0, 1.0)
        else:
            conf = (conf >= pivot).float()

    gamma = float(getattr(opt, "refscore_conf_gamma", 1.0))
    if gamma > 0.0 and gamma != 1.0:
        conf = conf.pow(gamma)

    min_conf = float(getattr(opt, "refscore_conf_min", 0.0))
    if min_conf > 0.0:
        conf = torch.where(conf >= min_conf, conf, torch.zeros_like(conf))

    return conf.detach()


def get_ref_score_confidence(cam, opt):
    if hasattr(cam, "ref_score_conf"):
        return prior_to_cuda(cam.ref_score_conf)
    if hasattr(cam, "ref_score"):
        return build_ref_score_confidence(cam.ref_score, opt)
    return None


def _normalize_adaptive_residual(residual, static_score, opt):
    """Robustly normalize full-model error only inside RefScore support."""
    valid = static_score > 0
    if not torch.any(valid):
        return torch.zeros_like(residual)

    values = residual[valid]
    low_q = float(getattr(opt, "adaptive_residual_low_quantile", 0.70))
    high_q = float(getattr(opt, "adaptive_residual_high_quantile", 0.95))
    low = torch.quantile(values, low_q)
    high = torch.quantile(values, high_q)
    if high <= low + 1e-6:
        return torch.zeros_like(residual)
    normalized = ((residual - low) / (high - low + 1e-6)).clamp(0.0, 1.0)
    return normalized * valid.float()


def _update_camera_adaptive_map(cam, residual, opt):
    """Update one camera's persistent error for adaptive supervision."""
    static_score = prior_to_cuda(cam.ref_score_static)
    residual_norm = _normalize_adaptive_residual(residual, static_score, opt)

    if hasattr(cam, "adaptive_error_ema"):
        old_error = prior_to_cuda(cam.adaptive_error_ema)
        ema = float(opt.adaptive_prior_ema)
        persistent_error = ema * old_error + (1.0 - ema) * residual_norm
    else:
        # Do not dilute the first observation with an implicit zero map.
        persistent_error = residual_norm

    persistent_error = persistent_error.clamp(0.0, 1.0)
    encoded_error = (persistent_error * 255.0).round().byte().cpu()
    cam.adaptive_error_ema = encoded_error
    cam.adaptive_difficulty = encoded_error.clone()

    # Camera-level score for reflection-aware hard-view replay. Use a robust
    # mean of joint reflection confidence and persistent reconstruction error;
    # isolated outlier pixels cannot dominate the sampling distribution.
    if hasattr(cam, "ref_score_conf_static"):
        ref_confidence = prior_to_cuda(cam.ref_score_conf_static).pow(1.5)
    else:
        ref_confidence = static_score.pow(1.5)
    joint_difficulty = ref_confidence * persistent_error
    valid_values = joint_difficulty[ref_confidence > 0]
    camera_score = 0.0
    if valid_values.numel() > 0:
        low_q = float(getattr(opt, "reflection_sampling_score_low_quantile", 0.70))
        high_q = float(getattr(opt, "reflection_sampling_score_high_quantile", 0.95))
        low = torch.quantile(valid_values, low_q)
        high = torch.quantile(valid_values, high_q)
        robust_values = valid_values[(valid_values >= low) & (valid_values <= high)]
        if robust_values.numel() > 0:
            camera_score = robust_values.mean().item()
    cam.reflection_sampling_score = max(float(camera_score), 0.0)

    # The successful Adaptive-Loss design leaves geometric coverage exactly
    # at the static RefScore. Non-unit bounds are supported only so the earlier
    # coverage ablation remains reproducible.
    floor = float(getattr(opt, "adaptive_prior_floor", 1.0))
    ceiling = float(getattr(opt, "adaptive_prior_ceiling", 1.0))
    if abs(floor - 1.0) > 1e-8 or abs(ceiling - 1.0) > 1e-8:
        modulation = floor + (ceiling - floor) * persistent_error
        effective_score = (static_score * modulation).clamp(0.0, 1.0)
        cam.ref_score = (effective_score * 255.0).round().byte().cpu()


def reflection_sampling_ratio(iteration, opt):
    """Scheduled probability of replaying a hard reflective training view."""
    if not getattr(opt, "use_reflection_view_sampling", False):
        return 0.0
    start = int(opt.reflection_sampling_start)
    peak_end = int(opt.reflection_sampling_peak_end)
    end = int(opt.reflection_sampling_end)
    maximum = float(opt.reflection_sampling_ratio)
    if iteration < start or iteration >= end:
        return 0.0
    if iteration < peak_end:
        return maximum
    decay = (iteration - peak_end) / max(end - peak_end, 1)
    return maximum * max(1.0 - decay, 0.0)


def sample_training_camera(viewpoint_stack, viewpoint_indices, all_cameras,
                           iteration, opt):
    """Mix coverage-preserving uniform sampling with hard-view replay."""
    ratio = reflection_sampling_ratio(iteration, opt)
    prioritized = False
    selected_score = 0.0

    if ratio > 0.0 and torch.rand((), device="cpu").item() < ratio:
        scores = torch.tensor([
            max(float(getattr(cam, "reflection_sampling_score", 0.0)), 0.0)
            for cam in all_cameras
        ], dtype=torch.float64)
        if scores.sum() > 0:
            temperature = max(float(opt.reflection_sampling_temperature), 1e-3)
            weights = (scores + 1e-12).pow(1.0 / temperature)
            selected = int(torch.multinomial(weights, 1).item())
            cam = all_cameras[selected]
            prioritized = True
            selected_score = float(scores[selected].item())
            # Deliberately do not pop from viewpoint_stack: prioritized samples
            # are additional replays, while uniform steps still cover every view.
            return cam, prioritized, selected_score, ratio

    idx = randint(0, len(viewpoint_indices) - 1)
    cam = viewpoint_stack.pop(idx)
    viewpoint_indices.pop(idx)
    selected_score = float(getattr(cam, "reflection_sampling_score", 0.0))
    return cam, prioritized, selected_score, ratio


def update_adaptive_ref_scores(cameras, gaussians, specular_mlp, pipe,
                               background, opt, iteration):
    """Legacy coverage-ablation update using full SH+ASG residuals.

    The validated Adaptive-Loss default does not call this extra render path.
    It remains available only when non-unit coverage bounds are explicitly set,
    so the negative coverage experiment can still be reproduced.
    """
    if not getattr(opt, 'use_ref_score', False) or not getattr(opt, 'use_adaptive_prior', False):
        return 0
    if iteration < opt.adaptive_prior_start:
        return 0
    if iteration >= getattr(opt, 'adaptive_prior_end', opt.densify_until_iter):
        return 0
    if iteration % opt.adaptive_prior_interval != 0:
        return 0

    updated = 0
    with torch.no_grad():
        for cam in cameras:
            if not hasattr(cam, 'ref_score_static'):
                continue

            # Measure what the complete model still gets wrong.  An SH-only
            # residual would keep marking highlights even after ASG learned them.
            xyz = gaussians.get_xyz
            viewdir = xyz - cam.camera_center
            viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-6)
            normal = gaussians.get_normal_axis(viewdir)
            mlp_color = specular_mlp.step(
                gaussians.get_asg_features, viewdir, normal.detach()
            )
            render_img = render_fastgs(
                cam,
                gaussians,
                pipe,
                background,
                opt.mult,
                mlp_color=mlp_color
            )["render"]

            gt = cam.original_image.cuda()
            residual = torch.abs(render_img - gt).mean(dim=0)
            _update_camera_adaptive_map(cam, residual, opt)
            updated += 1

    if updated:
        print(f"[Adaptive Prior] iter {iteration}: updated {updated} camera ref_score maps")
    return updated


# ============================================================
# TRAINING LOOP
# ============================================================

def training(dataset, opt, pipe):

    start_time = time.time()
    tb_writer = prepare_output_and_logger(dataset)

    # Reflection-aware sampling consumes the persistent scores maintained by
    # Adaptive Prior. Keep the public interface safe when the sampling flag is
    # used on its own.
    if (getattr(opt, "use_reflection_view_sampling", False)
            and not getattr(opt, "use_adaptive_prior", False)):
        opt.use_adaptive_prior = True
        print("[Reflection Sampling] enabling --use_adaptive_prior automatically")

    # Adaptive Prior is a refinement of Reflection Score.  Make the public
    # interface unambiguous: --use_adaptive_prior is sufficient to enable the
    # complete feature with its validated defaults.
    if getattr(opt, "use_adaptive_prior", False) and not opt.use_ref_score:
        opt.use_ref_score = True
        print("[Adaptive Prior] enabling --use_ref_score automatically")

    # ------------------------------------------------------------
    # INIT MODELS
    # ------------------------------------------------------------

    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    initial_gaussians = gaussians.get_xyz.shape[0]
    configure_refscore_budget(opt, initial_gaussians)
    gaussians.training_setup(opt)

    specular_mlp = SpecularModel(
        dataset.asg_degree,
        dataset.is_real,
        dataset.is_indoor,
        getattr(dataset, "asg_num_theta", -1),
        getattr(dataset, "asg_num_phi", -1),
        getattr(dataset, "specular_hidden", -1),
        getattr(dataset, "specular_layers", -1),
    )
    specular_mlp.train_setting(opt)

    # ------------------------------------------------------------
    # BG
    # ------------------------------------------------------------

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ------------------------------------------------------------
    # OPTIONAL LIVE WEB VIEWER
    # ------------------------------------------------------------
    web_viewer = None
    web_stats = []
    web_base_camera = None
    web_cameras = []
    web_target = None
    web_frames_dir = os.path.join(scene.model_path, "web_viewer_frames")
    web_live_dir = os.path.join(scene.model_path, "web_viewer_live")
    web_settings_path = os.path.join(scene.model_path, "web_viewer_settings.json")
    if getattr(opt, "web_viewer", False):
        viewer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web-viewer")
        if viewer_dir not in sys.path:
            sys.path.insert(0, viewer_dir)
        from server import ViewerServer
        web_viewer = ViewerServer(opt.web_host, opt.web_http_port, opt.web_ws_port)
        web_viewer.configure(
            interval=opt.web_stream_interval,
            save_frames=opt.web_save_frames,
        )
        web_viewer.start()
        web_cameras = scene.getTrainCameras()
        web_base_camera = web_cameras[0]
        web_target = gaussians.get_xyz.detach().mean(dim=0).cpu().numpy()
        os.makedirs(web_frames_dir, exist_ok=True)
        os.makedirs(web_live_dir, exist_ok=True)
        with open(os.path.join(scene.model_path, "web_viewer_manifest.json"), "w") as manifest_file:
            json.dump({
                "cameras": [
                    {"index": index, "image_name": camera.image_name}
                    for index, camera in enumerate(web_cameras)
                ]
            }, manifest_file, indent=2)

    # ------------------------------------------------------------
    # LOAD REFLECTION PRIORS (If available)
    # ------------------------------------------------------------
    import imageio
    ref_prior_dir = os.path.join(dataset.source_path, "reflection_prior")
    def load_prior_tensor(path, cam):
        prior_img = imageio.imread(path)
        if len(prior_img.shape) == 3:
            prior_img = prior_img[..., 0]
        prior_tensor = torch.as_tensor(prior_img, dtype=torch.uint8, device="cpu")
        if prior_tensor.shape[0] != cam.image_height or prior_tensor.shape[1] != cam.image_width:
            prior_tensor = torch.nn.functional.interpolate(
                prior_tensor.float().unsqueeze(0).unsqueeze(0),
                size=(cam.image_height, cam.image_width),
                mode='bilinear',
                align_corners=False
            ).squeeze().round().clamp(0, 255).byte()
        return prior_tensor.contiguous()

    if opt.use_ref_score and os.path.exists(ref_prior_dir):
        print("Loading Reflection Priors...")
        loaded_ref_priors = 0
        missing_ref_priors = 0
        for cam in scene.getTrainCameras():
            npath = os.path.join(ref_prior_dir, f"{cam.image_name}_ref_score.png")
            if os.path.exists(npath):
                try:
                    ref_tensor = load_prior_tensor(npath, cam)
                    cam.ref_score = ref_tensor
                    cam.ref_score_static = ref_tensor.clone()
                    conf_path = os.path.join(ref_prior_dir, f"{cam.image_name}_ref_conf.png")
                    if os.path.exists(conf_path):
                        cam.ref_score_conf = load_prior_tensor(conf_path, cam)
                    else:
                        conf = build_ref_score_confidence(ref_tensor, opt)
                        cam.ref_score_conf = (conf.clamp(0.0, 1.0) * 255.0).byte().cpu()
                    cam.ref_score_conf_static = cam.ref_score_conf.clone()
                    loaded_ref_priors += 1
                except Exception as error:
                    missing_ref_priors += 1
                    print(f"[Reflection Prior] Failed to load {npath}: {error}")
            else:
                missing_ref_priors += 1
        print(f"Loaded {loaded_ref_priors} reflection priors; missing {missing_ref_priors}.")
    elif opt.use_ref_score:
        print(f"Reflection prior enabled, but directory not found: {ref_prior_dir}")

    # ------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    save_iterations = {int(opt.iterations)}
    checkpoint_interval = max(0, int(getattr(opt, "checkpoint_interval", 0)))
    if checkpoint_interval:
        save_iterations.update(range(checkpoint_interval, opt.iterations + 1,
                                     checkpoint_interval))
    checkpoint_text = str(getattr(opt, "checkpoint_iterations", "") or "")
    for value in checkpoint_text.split(","):
        if value.strip():
            checkpoint = int(value.strip())
            if 1 <= checkpoint <= opt.iterations:
                save_iterations.add(checkpoint)
    print("Model checkpoints:", ", ".join(map(str, sorted(save_iterations))))

    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Training")
    ema_loss = 0.0

    # Cached boolean visibility mask for sparse ASG evaluation. This follows
    # the original fast path: reuse the previous frame's mask and fall back to
    # full ASG only when Gaussian count changes or an explicit refresh is due.
    prev_vis_mask: torch.Tensor | None = None
    asg_eval_count_total = 0
    asg_eval_steps = 0
    reflection_sampling_priority_steps = 0
    reflection_sampling_ratio_total = 0.0
    reflection_sampling_selected_score_total = 0.0
    reflection_sampling_active_steps = 0
    spec_reg_loss_total = 0.0
    spec_reg_steps = 0

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

        cam, prioritized_view, selected_view_score, sampling_ratio = sample_training_camera(
            viewpoint_stack,
            viewpoint_indices,
            scene.getTrainCameras(),
            iteration,
            opt,
        )
        if sampling_ratio > 0.0:
            reflection_sampling_active_steps += 1
            reflection_sampling_ratio_total += sampling_ratio
        if prioritized_view:
            reflection_sampling_priority_steps += 1
            reflection_sampling_selected_score_total += selected_view_score

        # --------------------------------------------------------
        # COMPUTE VIEWDIR + NORMAL
        # --------------------------------------------------------

        xyz = gaussians.get_xyz
        cam_center = cam.camera_center

        viewdir = xyz - cam_center
        viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-6)

        normal = gaussians.get_normal_axis(viewdir)

        # --------------------------------------------------------
        # SPECULAR (SG STYLE) — sparse MLP via previous-frame visibility
        # --------------------------------------------------------
        # Only Gaussians visible in the previous frame feed the ASG MLP.
        # Typically 10–30 % of Gaussians are on-screen per frame, so this
        # cuts MLP forward+backward cost by 3–10×.
        # After densification the count changes → we detect size mismatch
        # and fall back to all Gaussians for that one step.

        spec_sparse: torch.Tensor | None = None  # sparse MLP output [M, 3]
        vis_indices: torch.Tensor | None = None   # indices of evaluated Gaussians
        mlp_color: torch.Tensor | None = None     # full-scene buffer  [N, 3]

        if iteration > opt.specular_start_iter:
            n_gs = gaussians.get_xyz.shape[0]
            asg_feat = gaussians.get_asg_features  # [N, asg_degree]
            force_full_asg = (
                opt.full_asg_interval > 0 and
                iteration % opt.full_asg_interval == 0
            )

            # Determine which Gaussians to evaluate
            if not force_full_asg and prev_vis_mask is not None and prev_vis_mask.shape[0] == n_gs:
                vis_indices = prev_vis_mask.nonzero(as_tuple=False).squeeze(1)  # [M]
            else:
                # First specular step, full refresh, or count changed after densification.
                vis_indices = torch.arange(n_gs, device="cuda")
            asg_eval_count_total += int(vis_indices.numel())
            asg_eval_steps += 1

            if vis_indices.numel() > 0:
                spec_sparse = specular_mlp.step(
                    asg_feat[vis_indices],
                    viewdir[vis_indices],
                    normal.detach()[vis_indices],
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
            mlp_color=mlp_color,
        )

        image                 = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter     = render_pkg["visibility_filter"]
        radii                 = render_pkg["radii"]
        accum_metric_counts   = render_pkg["accum_metric_counts"]

        # ── Update cached visibility mask for the NEXT iteration ──────────
        # radii has shape [N]; (radii > 0) gives the boolean visibility mask.
        prev_vis_mask = (radii > 0)  # [N] bool

        # --------------------------------------------------------
        # LOSS
        # --------------------------------------------------------
        # Phase A design:
        #   • photometric_loss  — L1 + SSIM, unchanged from FastGS baseline.
        #     Gradients flow back through the renderer into mlp_color, so the
        #     specular MLP is already supervised by image reconstruction.
        #   • spec_reg          — lightweight L2 penalty on specular MLP outputs
        #     (Gaussian-space). Optional; prevents unbounded ASG energy/leakage,
        #     but can weaken only_asg if set too high.
        # --------------------------------------------------------

        gt = cam.original_image.cuda()

        adaptive_active = (
            getattr(opt, "use_adaptive_prior", False)
            and hasattr(cam, "ref_score_static")
            and iteration >= opt.adaptive_prior_start
            and iteration < getattr(opt, "adaptive_prior_end", opt.densify_until_iter)
        )
        if adaptive_active:
            # Reuse the current training render: this gives all training views
            # temporal coverage without an additional renderer invocation.
            with torch.no_grad():
                current_residual = torch.abs(image.detach() - gt).mean(dim=0)
                _update_camera_adaptive_map(cam, current_residual, opt)

        weight_map = None
        if adaptive_active and hasattr(cam, "adaptive_difficulty"):
            difficulty = prior_to_cuda(cam.adaptive_difficulty).unsqueeze(0)
            adaptive_strength = float(getattr(opt, "adaptive_loss_strength", 0.15))
            weight_map = 1.0 + adaptive_strength * difficulty

        if getattr(opt, "lambda_spec_l1_weight", 0.0) > 0.0 and hasattr(cam, "ref_score"):
            ref_w = get_ref_score_confidence(cam, opt).unsqueeze(0)
            spec_weight = 1.0 + opt.lambda_spec_l1_weight * ref_w
            weight_map = spec_weight if weight_map is None else weight_map * spec_weight

        if weight_map is not None:
            pixel_l1 = torch.abs(image - gt)
            # Keep the global loss scale stable; only redistribute gradients.
            weight_map = weight_map / weight_map.mean().clamp_min(1e-6)
            Ll1 = (pixel_l1 * weight_map).sum() / (3.0 * weight_map.sum().clamp_min(1e-6))
        else:
            Ll1 = l1_loss(image, gt)
        ssim_val = fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))
        photometric_loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim_val)
        )
        
        loss = photometric_loss
        if getattr(opt, "lambda_spec_reg", 0.0) > 0.0 and spec_sparse is not None:
            spec_reg_loss = (spec_sparse ** 2).mean()
            loss = loss + opt.lambda_spec_reg * spec_reg_loss
            spec_reg_loss_total += spec_reg_loss.detach().item()
            spec_reg_steps += 1

        loss.backward()

        # --- NO SH DEGREE RESTRICTION ---
        # Allow SH to train normally everywhere to form a smooth base color.
        # ASG will naturally handle the specular highlights due to gradient boosting.

        # --------------------------------------------------------
        # OPTIMIZER STEP
        # --------------------------------------------------------

        # Set skip_sh=False because we now use fine-grained gradient scaling per-Gaussian
        skip_sh = False
        gaussians.optimizer_step(
            iteration,
            skip_sh=skip_sh,
        )

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

            # --- GUIDED DENSIFICATION ---
            # Trả lại quyền kiểm soát sinh hạt cho cơ chế ADC của FastGS
            viewspace_grad = viewspace_point_tensor.grad.clone()
            
            class DummyTensor:
                def __init__(self, grad):
                    self.grad = grad
            
            dummy_viewspace = DummyTensor(viewspace_grad)

            gaussians.add_densification_stats(
                dummy_viewspace,
                visibility_filter
            )

            if (
                iteration > opt.densify_from_iter and
                iteration % opt.densification_interval == 0
            ):
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                score_camera_count = opt.num_score_cameras
                if getattr(opt, "use_adaptive_prior", False):
                    score_camera_count = getattr(
                        opt, "adaptive_prior_num_cameras", score_camera_count
                    )
                camlist = sampling_cameras(
                    scene.getTrainCameras().copy(), score_camera_count
                )

                # Adaptive Loss reuses the ordinary training render and keeps
                # static RefScore for geometry, so it needs no extra render here.
                # Only reproduce the legacy coverage ablation when explicitly
                # requested through non-unit modulation bounds.
                adaptive_coverage_enabled = (
                    getattr(opt, "use_adaptive_prior", False)
                    and (
                        abs(float(getattr(opt, "adaptive_prior_floor", 1.0)) - 1.0) > 1e-8
                        or abs(float(getattr(opt, "adaptive_prior_ceiling", 1.0)) - 1.0) > 1e-8
                    )
                )
                with torch.no_grad():
                    if adaptive_coverage_enabled:
                        update_adaptive_ref_scores(
                            camlist, gaussians, specular_mlp, pipe, background,
                            opt, iteration
                        )
                    importance_score, pruning_score = compute_gaussian_score_fastgs(
                        camlist, gaussians, pipe, background, opt, DENSIFY=True, iteration=iteration
                    )

                gaussians.densify_and_prune_fastgs(
                    max_screen_size=size_threshold,
                    min_opacity=0.005,
                    extent=scene.cameras_extent,
                    radii=radii,
                    args=opt,
                    importance_score=importance_score,
                    pruning_score=pruning_score
                )

            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                gaussians.reset_opacity()

        # The multiview consistent pruning of fastgs. We do it every 3k iterations after 15k
        if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
            camlist = sampling_cameras(scene.getTrainCameras().copy(), opt.num_score_cameras)
            with torch.no_grad():
                _, pruning_score = compute_gaussian_score_fastgs(
                    camlist, gaussians, pipe, background, opt, False, iteration=iteration
                )
            gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)

        # --------------------------------------------------------
        # LIVE VIEWER + PER-ITERATION TELEMETRY
        # --------------------------------------------------------
        if web_viewer is not None:
            settings = _merged_web_settings(web_viewer, web_settings_path)
            while settings["paused"]:
                time.sleep(0.05)
                settings = _merged_web_settings(web_viewer, web_settings_path)

            torch.cuda.synchronize()
            allocated_mib = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved_mib = torch.cuda.memory_reserved() / (1024 ** 2)
            gaussian_count = int(gaussians.get_xyz.shape[0])
            web_stats.append({
                "iteration": iteration,
                "gaussian_count": gaussian_count,
                "vram_allocated_mib": round(allocated_mib, 2),
                "vram_reserved_mib": round(reserved_mib, 2),
            })

            interval = max(1, int(settings["interval"]))
            record_interval = max(1, int(settings.get("record_interval", 50)))
            record_due = (bool(settings.get("save_frames", False))
                          and iteration % record_interval == 0)
            should_render_web = iteration % interval == 0 or record_due
            if should_render_web:
                camera_index = max(0, min(
                    len(web_cameras) - 1, int(settings.get("camera_index", 0))
                ))
                web_base_camera = web_cameras[camera_index]
                with torch.no_grad():
                    rgb_jpeg, geometry_jpeg, timeline_frames = _render_web_pair(
                        web_base_camera, web_target, settings,
                        opt.web_width, opt.web_height, gaussians,
                        specular_mlp, pipe, background, opt.mult,
                        iteration > opt.specular_start_iter,
                        capture_all=record_due,
                    )

                web_viewer.publish(iteration, gaussian_count, allocated_mib,
                                   reserved_mib, rgb_jpeg, geometry_jpeg)
                _write_web_live(
                    web_live_dir, iteration, gaussian_count, allocated_mib,
                    reserved_mib, rgb_jpeg, geometry_jpeg,
                )
                if record_due and timeline_frames:
                    for name, encoded in timeline_frames.items():
                        folder = os.path.join(web_frames_dir, name)
                        os.makedirs(folder, exist_ok=True)
                        with open(os.path.join(folder, f"{iteration:06d}.jpg"), "wb") as frame_file:
                            frame_file.write(encoded)

            if iteration % 100 == 0 or iteration == opt.iterations:
                with open(os.path.join(scene.model_path, "web_viewer_stats.json"), "w") as stats_file:
                    json.dump({
                        "camera_image": web_base_camera.image_name,
                        "samples": web_stats,
                    }, stats_file, indent=2)

        # --------------------------------------------------------
        # SAVE
        # --------------------------------------------------------

        if iteration in save_iterations:
            print(f"[ITER {iteration}] Saving...")
            scene.save(iteration)

            # ✅ QUAN TRỌNG NHẤT
            specular_mlp.save_weights(scene.model_path, iteration)

    # ------------------------------------------------------------
    # SAVE METADATA
    # ------------------------------------------------------------
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    avg_asg_eval_count = asg_eval_count_total / asg_eval_steps if asg_eval_steps > 0 else 0.0
    avg_reflection_sampling_ratio = (
        reflection_sampling_ratio_total / reflection_sampling_active_steps
        if reflection_sampling_active_steps > 0 else 0.0
    )
    avg_prioritized_view_score = (
        reflection_sampling_selected_score_total / reflection_sampling_priority_steps
        if reflection_sampling_priority_steps > 0 else 0.0
    )
    avg_spec_reg_loss = spec_reg_loss_total / spec_reg_steps if spec_reg_steps > 0 else 0.0
    heatmap_path = save_gaussian_view_heatmaps(
        scene.getTestCameras(),
        gaussians,
        scene.model_path,
        opt.iterations,
        render_fastgs,
        (pipe, background, opt.mult),
    )
    if heatmap_path:
        print(f"Saved Gaussian distribution heatmaps to {heatmap_path}")
    metadata = {
        "scene": dataset.source_path.split("/")[-1],
        "git_branch": get_git_branch(),
        "image_scale": dataset.images,
        "iterations": opt.iterations,
        "saved_checkpoints": sorted(save_iterations),
        "checkpoint_interval": checkpoint_interval,
        "initial_gaussians": initial_gaussians,
        "final_gaussians": gaussians.get_xyz.shape[0],
        "gaussian_heatmap_dir": os.path.relpath(heatmap_path, dataset.model_path) if heatmap_path else None,
        "training_time_seconds": round(duration, 2),
        "training_time_formatted": f"{minutes}m {seconds}s",
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "asg_degree": dataset.asg_degree,
        "asg_num_theta": getattr(dataset, "asg_num_theta", -1),
        "asg_num_phi": getattr(dataset, "asg_num_phi", -1),
        "specular_hidden": getattr(dataset, "specular_hidden", -1),
        "specular_layers": getattr(dataset, "specular_layers", -1),
        "specular_start_iter": opt.specular_start_iter,
        "full_asg_interval": opt.full_asg_interval,
        "avg_asg_eval_count": round(avg_asg_eval_count, 2),
        "num_score_cameras": opt.num_score_cameras,
        "use_ref_score": opt.use_ref_score,
        "max_refscore_gaussians": opt.max_refscore_gaussians,
        "refscore_budget_multiplier": opt.refscore_budget_multiplier,
        "refscore_budget_min": opt.refscore_budget_min,
        "refscore_budget_max": opt.refscore_budget_max,
        "refscore_decay_power": opt.refscore_decay_power,
        "refscore_min_strength": opt.refscore_min_strength,
        "refscore_threshold_min": opt.refscore_threshold_min,
        "refscore_threshold_max": opt.refscore_threshold_max,
        "refscore_strength": opt.refscore_strength,
        "refscore_conf_quantile": opt.refscore_conf_quantile,
        "refscore_conf_gamma": opt.refscore_conf_gamma,
        "refscore_conf_min": opt.refscore_conf_min,
        "use_adaptive_prior": opt.use_adaptive_prior,
        "adaptive_prior_start": opt.adaptive_prior_start,
        "adaptive_prior_end": opt.adaptive_prior_end,
        "adaptive_prior_interval": opt.adaptive_prior_interval,
        "adaptive_prior_num_cameras": opt.adaptive_prior_num_cameras,
        "adaptive_prior_ema": opt.adaptive_prior_ema,
        "adaptive_prior_floor": opt.adaptive_prior_floor,
        "adaptive_prior_ceiling": opt.adaptive_prior_ceiling,
        "adaptive_residual_low_quantile": opt.adaptive_residual_low_quantile,
        "adaptive_residual_high_quantile": opt.adaptive_residual_high_quantile,
        "adaptive_loss_strength": opt.adaptive_loss_strength,
        "ref_prior_method": opt.ref_prior_method,
        "ti_thresh": opt.ti_thresh,
        "ti_bright": opt.ti_bright,
        "sk_intensity": opt.sk_intensity,
        "sk_saturation": opt.sk_saturation,
        "ref_conf_gamma": opt.ref_conf_gamma,
        "ref_conf_quantile": opt.ref_conf_quantile,
        "ref_conf_smooth_radius": opt.ref_conf_smooth_radius,
        "f_rest_warmup_until": opt.f_rest_warmup_until,
        "f_rest_interval_early": opt.f_rest_interval_early,
        "f_rest_interval_mid": opt.f_rest_interval_mid,
        "f_rest_interval_late": opt.f_rest_interval_late,
        "use_reflection_view_sampling": opt.use_reflection_view_sampling,
        "reflection_sampling_ratio": opt.reflection_sampling_ratio,
        "reflection_sampling_start": opt.reflection_sampling_start,
        "reflection_sampling_peak_end": opt.reflection_sampling_peak_end,
        "reflection_sampling_end": opt.reflection_sampling_end,
        "reflection_sampling_temperature": opt.reflection_sampling_temperature,
        "reflection_sampling_score_low_quantile": opt.reflection_sampling_score_low_quantile,
        "reflection_sampling_score_high_quantile": opt.reflection_sampling_score_high_quantile,
        "reflection_sampling_priority_steps": reflection_sampling_priority_steps,
        "reflection_sampling_active_steps": reflection_sampling_active_steps,
        "avg_reflection_sampling_ratio": round(avg_reflection_sampling_ratio, 6),
        "avg_prioritized_view_score": round(avg_prioritized_view_score, 8),
        "lambda_spec_l1_weight": opt.lambda_spec_l1_weight,
        "lambda_spec_reg": opt.lambda_spec_reg,
        "avg_spec_reg_loss": round(avg_spec_reg_loss, 8),
        "datetime_completed": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    info_path = os.path.join(dataset.model_path, "train_info.json")
    with open(info_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Training metadata saved to {info_path}")

    # ------------------------------------------------------------
    # AUTOMATIC TEST RENDER + METRICS
    # ------------------------------------------------------------
    evaluation_status_path = os.path.join(scene.model_path, "evaluation_status.json")

    def write_evaluation_status(status, phase, detail=""):
        temporary = evaluation_status_path + ".tmp"
        with open(temporary, "w") as status_file:
            json.dump({"status": status, "phase": phase, "detail": detail,
                       "iteration": opt.iterations}, status_file, indent=2)
        os.replace(temporary, evaluation_status_path)

    test_cameras = scene.getTestCameras()
    if web_viewer is not None and test_cameras:
        try:
            write_evaluation_status("running", "Rendering test views",
                                    f"0 / {len(test_cameras)} views")
            print(f"Automatic evaluation: rendering {len(test_cameras)} test views...")
            from render import render_set, save_fps_to_results
            test_fps = render_set(
                scene.model_path, "test", opt.iterations, test_cameras,
                gaussians, pipe, background, specular_mlp, opt
            )
            save_fps_to_results(scene.model_path, opt.iterations, test_fps)
            write_evaluation_status("running", "Computing metrics",
                                    "PSNR · SSIM · LPIPS · specular diagnostics")
            from metrics import evaluate
            evaluate([scene.model_path])
            results_path = os.path.join(scene.model_path, "results.json")
            if not os.path.isfile(results_path):
                raise RuntimeError("metrics.py did not produce results.json")
            with open(results_path, "r") as results_file:
                evaluation_results = json.load(results_file)
            method_results = evaluation_results.get(f"ours_{opt.iterations}", {})
            if not all(name in method_results for name in ("PSNR", "SSIM", "LPIPS")):
                raise RuntimeError("metrics.py did not produce PSNR/SSIM/LPIPS")
            write_evaluation_status("complete", "Evaluation complete",
                                    f"{len(test_cameras)} test views")
            print("Automatic evaluation complete.")
        except Exception as evaluation_error:
            write_evaluation_status("failed", "Evaluation failed",
                                    str(evaluation_error))
            print("Automatic evaluation failed:", evaluation_error)
    elif web_viewer is not None:
        write_evaluation_status(
            "skipped", "Evaluation skipped",
            "No test cameras. Enable Evaluation split before starting training."
        )

    # Keep the CUDA scene alive as an interactive final-result viewer.  The
    # persistent launcher can still terminate this process with its Stop button.
    if web_viewer is not None:
        print("Training complete. Final web viewer remains interactive; use Stop to close it.")
        final_target = gaussians.get_xyz.detach().mean(dim=0).cpu().numpy()
        current_view_iteration = opt.iterations
        last_view_signature = None
        while True:
            settings = _merged_web_settings(web_viewer, web_settings_path)
            if settings.get("close_viewer", False):
                break
            camera_index = max(0, min(
                len(web_cameras) - 1, int(settings.get("camera_index", 0))
            ))
            requested_iteration = int(settings.get("checkpoint_iteration", -1))
            if requested_iteration < 0:
                requested_iteration = opt.iterations
            if requested_iteration != current_view_iteration:
                point_dir = os.path.join(
                    scene.model_path, "point_cloud", f"iteration_{requested_iteration}"
                )
                spec_dir = os.path.join(
                    scene.model_path, "specular", f"iteration_{requested_iteration}"
                )
                ply_path = os.path.join(point_dir, "point_cloud.ply")
                asg_path = os.path.join(point_dir, "asg.pt")
                spec_path = os.path.join(spec_dir, "specular.pth")
                if all(os.path.isfile(path) for path in (ply_path, asg_path, spec_path)):
                    gaussians.load_ply(ply_path)
                    gaussians._features_asg = torch.load(
                        asg_path, map_location="cuda"
                    ).cuda()
                    specular_mlp.load_weights(
                        scene.model_path, iteration=requested_iteration
                    )
                    current_view_iteration = requested_iteration
                    final_target = gaussians.get_xyz.detach().mean(dim=0).cpu().numpy()
                    last_view_signature = None
                else:
                    requested_iteration = current_view_iteration
            signature = (
                requested_iteration, str(settings.get("rgb_component", "render")),
                camera_index, float(settings.get("yaw", 0.0)),
                float(settings.get("pitch", 0.0)), float(settings.get("roll", 0.0)),
                float(settings.get("zoom", 1.0)), float(settings.get("fov_scale", 1.2)),
                float(settings.get("splat_scale", 1.35)),
                float(settings.get("geometry_opacity", 0.72)),
            )
            if signature != last_view_signature:
                web_base_camera = web_cameras[camera_index]
                torch.cuda.synchronize()
                allocated_mib = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved_mib = torch.cuda.memory_reserved() / (1024 ** 2)
                gaussian_count = int(gaussians.get_xyz.shape[0])
                with torch.no_grad():
                    rgb_jpeg, geometry_jpeg, _ = _render_web_pair(
                        web_base_camera, final_target, settings,
                        opt.web_width, opt.web_height, gaussians,
                        specular_mlp, pipe, background, opt.mult,
                        current_view_iteration > opt.specular_start_iter,
                    )
                web_viewer.publish(
                    current_view_iteration, gaussian_count, allocated_mib,
                    reserved_mib, rgb_jpeg, geometry_jpeg,
                )
                _write_web_live(
                    web_live_dir, current_view_iteration, gaussian_count, allocated_mib,
                    reserved_mib, rgb_jpeg, geometry_jpeg, phase="final",
                )
                last_view_signature = signature
            time.sleep(0.05)

# ============================================================
# UTILS
# ============================================================

def get_git_branch():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def prepare_output_and_logger(args):

    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
    else:
        # If the output directory already exists and contains files, back it up first
        if os.path.exists(args.model_path) and os.listdir(args.model_path):
            import datetime
            existing_branch = "unknown"
            info_file = os.path.join(args.model_path, "train_info.json")
            if os.path.exists(info_file):
                try:
                    with open(info_file, "r") as f:
                        old_info = json.load(f)
                        existing_branch = old_info.get("git_branch", "unknown")
                except Exception:
                    pass
            
            if existing_branch == "unknown":
                existing_branch = get_git_branch()
                
            # Use last modified time of the directory for the backup timestamp
            mtime = os.path.getmtime(args.model_path)
            timestamp = datetime.datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")

            parent_dir = os.path.dirname(args.model_path)
            folder_name = os.path.basename(args.model_path.rstrip('/'))
            backup_dir = os.path.join(parent_dir, "backups", folder_name)
            backup_path = os.path.join(backup_dir, f"{existing_branch}_{timestamp}")
            
            print(f"Output folder already exists and is not empty. Moving old run to: {backup_path}")
            try:
                os.makedirs(backup_dir, exist_ok=True)
                os.rename(args.model_path, backup_path)
            except Exception as e:
                print(f"Warning: Could not rename existing output folder: {e}")

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
