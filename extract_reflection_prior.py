# ============================================================
# Training (Spec-Gaussian + FastGS)
# ============================================================

import torch
import numpy as np
import os, time, sys, json
from random import randint
from tqdm import tqdm
import uuid

from fused_ssim import fused_ssim as fast_ssim
from utils.loss_utils import l1_loss
from utils.image_utils import psnr
from torchvision.utils import save_image
import imageio
from collections import defaultdict
from utils.graphics_utils import patch_offsets, patch_warp

from gaussian_renderer import render_fastgs
from scene import Scene, GaussianModel, SpecularModel

from utils.general_utils import safe_state
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except:
    TENSORBOARD_FOUND = False


# ============================================================
# TRAINING LOOP
# ============================================================

def training(dataset, opt, pipe):

    start_time = time.time()
    tb_writer = prepare_output_and_logger(dataset)

    # ------------------------------------------------------------
    # INIT MODELS
    # ------------------------------------------------------------

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    initial_gaussians = gaussians.get_xyz.shape[0]
    gaussians.training_setup(opt)

    specular_mlp = SpecularModel(dataset.is_real, dataset.is_indoor)
    specular_mlp.train_setting(opt)

    # ------------------------------------------------------------
    # BG
    # ------------------------------------------------------------

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Train for 5000 iterations to get geometry
    extract_iters = 5000
    progress_bar = tqdm(range(1, extract_iters + 1), desc="Training for Geometry")
    
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        idx = randint(0, len(viewpoint_indices) - 1)
        cam = viewpoint_stack.pop(idx)
        viewpoint_indices.pop(idx)

        render_pkg = render_fastgs(cam, gaussians, pipe, background, opt.mult)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt = cam.original_image.cuda()
        Ll1 = l1_loss(image, gt)
        ssim_val = fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
        loss.backward()
        gaussians.optimizer_step(iteration)

        if iteration < opt.densify_until_iter:
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                camlist = sampling_cameras(scene.getTrainCameras().copy())
                with torch.no_grad():
                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, background, opt, DENSIFY=True)
                gaussians.densify_and_prune_fastgs(max_screen_size=size_threshold, min_opacity=0.005, extent=scene.cameras_extent, radii=radii, args=opt, importance_score=importance_score, pruning_score=pruning_score)

    print("Geometry formed. Computing Reflection Scores...")
    calc_ref_score(scene, gaussians, pipe, background, dataset)
    print("Prior extraction complete!")

    # ------------------------------------------------------------
    # SAVE METADATA
    # ------------------------------------------------------------
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    metadata = {
        "scene": dataset.source_path.split("/")[-1],
        "git_branch": get_git_branch(),
        "image_scale": dataset.images,
        "iterations": opt.iterations,
        "initial_gaussians": initial_gaussians,
        "final_gaussians": gaussians.get_xyz.shape[0],
        "training_time_seconds": round(duration, 2),
        "training_time_formatted": f"{minutes}m {seconds}s",
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "datetime_completed": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    info_path = os.path.join(dataset.model_path, "train_info.json")
    with open(info_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Training metadata saved to {info_path}")

# ============================================================
# UTILS
# ============================================================

def get_multi_view_neighbor(scene:Scene):
    nearest_neighbor_dicts = defaultdict()
    multi_view_num = 10
    multi_view_max_angle = 90
    multi_view_min_angle = 5
    multi_view_min_dis = 0.1
    multi_view_max_dis = 1.5
        
    world_view_transforms = []
    camera_centers = []
    center_rays = []
    for cur_idx, cur_cam in enumerate(scene.getTrainCameras().copy()):
        world_view_transforms.append(cur_cam.world_view_transform)
        camera_centers.append(cur_cam.camera_center)
        cur_cam = scene.getTrainCameras()[cur_idx]
        if isinstance(cur_cam.R, torch.Tensor):
            R = cur_cam.R.float().cuda()
            T = cur_cam.T.float().cuda()
        else:
            R = torch.tensor(cur_cam.R).float().cuda()
            T = torch.tensor(cur_cam.T).float().cuda()
        center_ray = torch.tensor([0.0,0.0,1.0]).float().cuda()
        center_ray = center_ray@R.transpose(-1,-2)
        center_rays.append(center_ray)
        
    world_view_transforms = torch.stack(world_view_transforms)
    camera_centers = torch.stack(camera_centers, dim=0)
    center_rays = torch.stack(center_rays, dim=0)
    center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
    diss = torch.norm(camera_centers[:,None] - camera_centers[None], dim=-1).detach().cpu().numpy()
    tmp = torch.sum(center_rays[:,None]*center_rays[None], dim=-1).clamp(-1,1)
    angles = torch.arccos(tmp)*180/3.14159
    angles = angles.detach().cpu().numpy()
    
    for id, cur_cam in enumerate(scene.getTrainCameras().copy()):
        sorted_indices = np.lexsort((angles[id], diss[id]))
        mask = (angles[id][sorted_indices] < multi_view_max_angle) & \
            (diss[id][sorted_indices] > multi_view_min_dis) & \
            (diss[id][sorted_indices] < multi_view_max_dis) & \
            (angles[id][sorted_indices] > multi_view_min_angle)
        sorted_indices = sorted_indices[mask]
        nearest_neighbor_dicts[cur_cam.image_name] = [
            (index,scene.getTrainCameras()[index].image_name) for index in sorted_indices[:multi_view_num]
        ]
    return nearest_neighbor_dicts

@torch.no_grad()
def calc_ref_score(scene:Scene, gaussians:GaussianModel, pipe, bg, dataset):
    patch_size = 4
    total_patch_size = (2 * patch_size + 1) ** 2
    
    depth_map_cached = defaultdict()
    viewpoint_stack = scene.getTrainCameras().copy()
    
    print("Caching depths...")
    for viewpoint_cam in tqdm(viewpoint_stack):
        image_name = viewpoint_cam.image_name
        
        # HACK: Render depth map by passing Z-coordinates as color
        xyz = gaussians.get_xyz
        view_xyz = xyz @ viewpoint_cam.world_view_transform[:3, :3] + viewpoint_cam.world_view_transform[3, :3]
        depths = view_xyz[:, 2].unsqueeze(1).repeat(1, 3) # [N, 3]
        
        # Render using depth as color
        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, 1.0, override_color=depths)
        
        # The rendered image is now the surf_depth map!
        depth_map_cached[image_name] = render_pkg["render"][0:1].detach().cpu() # Shape [1, H, W]
        
    nearest_neighbor_dicts = get_multi_view_neighbor(scene)
    save_dir = os.path.join(dataset.source_path, "ref_priors")
    os.makedirs(save_dir, exist_ok=True)
    
    for viewpoint_cam in tqdm(viewpoint_stack, desc="Computing Reflection Score"):
        ref_image_name = viewpoint_cam.image_name
        ref_depth:torch.Tensor = depth_map_cached[ref_image_name].cuda()
        H, W = ref_depth.squeeze().shape

        nearest_cam_lst = [viewpoint_stack[idx] for (idx,_) in nearest_neighbor_dicts[ref_image_name]]
        
        ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
        pixels = torch.stack([ix, iy], dim=-1).float().to(ref_depth.device)

        pts = gaussians.get_points_from_depth(viewpoint_cam, ref_depth)
        
        sampled_rgb_lst = []
        anchored_rgb = None
        
        for nearest_cam in nearest_cam_lst:
            pts_in_nearest_cam = pts.clone() @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
            
            map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, depth_map_cached[nearest_cam.image_name].cuda(), pts_in_nearest_cam)
            pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3] + 1e-6)
            pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
            
            if isinstance(nearest_cam.R, torch.Tensor):
                R = nearest_cam.R.float().cuda()
                T = nearest_cam.T.float().cuda()
            else:
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
            
            pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
            pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
            pts_projections = torch.stack([
                pts_in_view_cam[:,0] * viewpoint_cam.Fx / (pts_in_view_cam[:,2] + 1e-6) + viewpoint_cam.Cx,
                pts_in_view_cam[:,1] * viewpoint_cam.Fy / (pts_in_view_cam[:,2] + 1e-6) + viewpoint_cam.Cy
            ], -1).float()
            
            pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
            d_mask = d_mask & (pixel_noise < 2.0)
            
            d_mask = d_mask.reshape(-1)
            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask] 
            pixels_this_neighbor = pixels.clone().reshape(-1,2)[valid_indices]
            offsets = patch_offsets(patch_size, pixels_this_neighbor.device)
            ori_pixels_patch = pixels_this_neighbor.reshape(-1, 1, 2) + offsets.float()

            if anchored_rgb is None:
                ref_pixels_patch = pixels.clone().reshape(-1,1,2) + offsets.float()
                ref_pixels_patch[:, :, 0] = 2 * ref_pixels_patch[:, :, 0] / (W - 1) - 1.0
                ref_pixels_patch[:, :, 1] = 2 * ref_pixels_patch[:, :, 1] / (H - 1) - 1.0
                ref_image_rgb = viewpoint_cam.original_image.cuda()
                anchored_rgb = torch.nn.functional.grid_sample(ref_image_rgb.unsqueeze(0), ref_pixels_patch.reshape(1,-1,1,2), align_corners=True)
                anchored_rgb = anchored_rgb.reshape(3,-1,total_patch_size)

            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

            # Simplified normal using viewdir for quick approximation
            ref_local_n = torch.tensor([0,0,1], dtype=torch.float32, device="cuda").expand(pts.shape[0], 3)[valid_indices]
            ref_local_d = (pts[:,2])[valid_indices]

            H_ref_to_neareast = ref_to_neareast_r[None] - torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1)) / (ref_local_d[...,None,None]+1e-6)
            
            # Manual K matrix construction
            K_nearest = torch.tensor([[nearest_cam.Fx, 0, nearest_cam.Cx], [0, nearest_cam.Fy, nearest_cam.Cy], [0, 0, 1]], dtype=torch.float32, device="cuda")
            K_inv_view = torch.tensor([[1/viewpoint_cam.Fx, 0, -viewpoint_cam.Cx/viewpoint_cam.Fx], [0, 1/viewpoint_cam.Fy, -viewpoint_cam.Cy/viewpoint_cam.Fy], [0, 0, 1]], dtype=torch.float32, device="cuda")
            
            H_ref_to_neareast = torch.matmul(K_nearest[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
            H_ref_to_neareast = H_ref_to_neareast @ K_inv_view

            grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
            grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
            grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
            image_rgb = nearest_cam.original_image.cuda()
            
            sampled_rgb = torch.nn.functional.grid_sample(image_rgb.unsqueeze(0), grid.reshape(1,-1,1,2), align_corners=True)
            sampled_rgb = sampled_rgb.reshape(3,-1,total_patch_size)
            
            sampled_rgb_map = torch.zeros((3,H,W,total_patch_size)).to(sampled_rgb.device)
            sampled_rgb_map[:,pixels_this_neighbor[:,1].long(),pixels_this_neighbor[:,0].long(),:] = sampled_rgb
            sampled_rgb_lst += [sampled_rgb_map.reshape((3,-1,total_patch_size)).cpu()]
        
        if len(sampled_rgb_lst) == 0:
            continue
            
        all_sampled_rgb = torch.stack(sampled_rgb_lst).permute((0,2,3,1)).contiguous() # CPU
        anchored_rgb = anchored_rgb.unsqueeze(0).permute((0,2,3,1)).contiguous().cpu()
        
        try:
            tot_sampled_rgb = torch.concat([all_sampled_rgb,anchored_rgb],dim=0)
            mean_color = tot_sampled_rgb.mean(0,keepdim=True)
            std_color = (((tot_sampled_rgb-mean_color)**2)).mean(0)**0.5
            RS_temp = std_color.sum(-1).mean(-1)
            
            # Normalize and save
            RS_temp_vis = RS_temp.reshape((H,W))
            if RS_temp_vis.max() > 0:
                RS_temp_vis = RS_temp_vis / RS_temp_vis.max()
                
            RS_temp_vis = (RS_temp_vis * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            imageio.imwrite(f"{save_dir}/{ref_image_name}_ref_score.png", RS_temp_vis)
        except Exception as e:
            print("Error computing variance:", e)

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

