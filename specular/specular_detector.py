import torch
import math
from gaussian_renderer import render_fastgs
from utils.fast_utils import sampling_cameras

def pixel_to_world_ray(camera, x, y):
    W = camera.image_width
    H = camera.image_height

    ndc_x = (x + 0.5) / W * 2.0 - 1.0
    ndc_y = (y + 0.5) / H * 2.0 - 1.0
    ndc_y = -ndc_y

    tan_half_fovx = math.tan(camera.FoVx * 0.5)
    tan_half_fovy = math.tan(camera.FoVy * 0.5)

    ray_cam = torch.tensor(
        [
            ndc_x * tan_half_fovx,
            ndc_y * tan_half_fovy,
            1.0
        ],
        device="cuda"
    )
    ray_cam = ray_cam / torch.norm(ray_cam)

    R = camera.world_view_transform[:3, :3].T
    ray_dir = R @ ray_cam
    ray_dir = ray_dir / torch.norm(ray_dir)

    ray_origin = camera.camera_center
    return ray_origin, ray_dir

def gaussian_ray_distance(gaussians_xyz, ray_origin, ray_dir):
    """
    Compute distance from all Gaussians to a ray.
    gaussians_xyz: [N, 3]
    ray_origin: [3]
    ray_dir: normalized [3]
    """
    v = gaussians_xyz - ray_origin
    cross = torch.cross(v, ray_dir)
    return torch.norm(cross, dim=1)

# --------------------------------------------------
# 1. Pixel-level spotlight detection
# --------------------------------------------------
def detect_pixel_spotlights(render_img, gt_img, topk_ratio=0.001):
    """
    Detect pixel-level specular hotspots.

    Args:
        render_img: Tensor [3, H, W]
        gt_img:     Tensor [3, H, W]

    Returns:
        hotspot_mask: BoolTensor [H, W]
    """
    # L1 error per pixel
    pixel_err = torch.mean(torch.abs(render_img - gt_img), dim=0)  # [H, W]

    # Top-k pixels (peak-based)
    flat = pixel_err.view(-1)
    k = max(1, int(topk_ratio * flat.numel()))
    thresh = torch.topk(flat, k=k, largest=True).values.min()

    hotspot_mask = pixel_err >= thresh
    return hotspot_mask


# --------------------------------------------------
# 2. Pixel -> Gaussian assignment
# --------------------------------------------------
def assign_pixels_to_gaussians_by_ray(
    hotspot_mask,
    camera,
    gaussians_xyz,
    max_dist=0.05
):
    """
    hotspot_mask: [H, W] bool
    gaussians_xyz: [N, 3]

    Returns:
        gaussian_votes: [N]
    """
    N = gaussians_xyz.shape[0]
    votes = torch.zeros(N, device="cuda")

    ys, xs = torch.where(hotspot_mask)

    for y, x in zip(ys, xs):
        ray_o, ray_d = pixel_to_world_ray(camera, x.item(), y.item())

        # Compute distance to all Gaussians
        dists = torch.norm(
            torch.cross(gaussians_xyz - ray_o, ray_d),
            dim=1
        )

        g = torch.argmin(dists)
        if dists[g] < max_dist:
            votes[g] += 1

    return votes

# --------------------------------------------------
# 3. Orchestrator
# --------------------------------------------------
@torch.no_grad()
def detect_specular_gaussians(
    scene,
    gaussians,
    pipeline,
    background,
    mult,
    topk_ratio=0.001,
    min_pixel_count=10,
    max_ray_dist=0.05
):
    """
    Pixel-space spotlight detection + ray-based Gaussian assignment
    """

    device = gaussians.get_xyz.device
    gaussians_xyz = gaussians.get_xyz
    num_gaussians = gaussians_xyz.shape[0]

    gaussian_votes = torch.zeros(num_gaussians, device=device)

    cameras = sampling_cameras(scene.getTrainCameras().copy())
    cameras = cameras[:8]  # đủ để thấy view-dependence

    for cam in cameras:
        pkg = render_fastgs(
            cam,
            gaussians,
            pipeline,
            background,
            mult
        )

        render_img = pkg["render"]
        gt_img = cam.original_image[:3].to(device)

        # ---- Pixel-level spotlight detection ----
        pixel_err = torch.mean(torch.abs(render_img - gt_img), dim=0)
        flat = pixel_err.view(-1)

        k = max(1, int(topk_ratio * flat.numel()))
        thresh = torch.topk(flat, k=k).values.min()
        hotspot_mask = pixel_err >= thresh

        ys, xs = torch.where(hotspot_mask)

        # ---- Ray -> Gaussian assignment ----
        for y, x in zip(ys, xs):
            ray_o, ray_d = pixel_to_world_ray(cam, x.item(), y.item())
            dists = gaussian_ray_distance(gaussians_xyz, ray_o, ray_d)

            g = torch.argmin(dists)
            if dists[g] < max_ray_dist:
                gaussian_votes[g] += 1

    specular_mask = gaussian_votes >= min_pixel_count

    print(
        f"[Specular-Detect] {specular_mask.float().mean()*100:.2f}% "
        f"Gaussians marked as specular"
    )

    return specular_mask