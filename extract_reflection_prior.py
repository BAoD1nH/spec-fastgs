# ============================================================
# Extract Reflection Prior (Tan-Ikeuchi Algorithm)
# ============================================================

import os
import time
import imageio
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams

def tan_ikeuchi_score(img01, thresh=0.35, bright_floor=0.6):
    """
    Tan & Ikeuchi (2005) specular-free image logic.
    img01: [H, W, 3] float array in [0, 1]
    Returns mask and score.
    """
    Imin = img01.min(axis=-1)
    Imax = img01.max(axis=-1)
    score = Imin
    mask = (score > thresh) & (Imax > bright_floor)
    
    # We'll use the masked score directly as the Ref Score.
    # Where mask is false, score is 0.
    final_score = np.where(mask, score, 0.0)
    
    # Normalize to [0, 1] if there are any specular pixels
    if final_score.max() > 0:
        final_score = final_score / final_score.max()
        
    return final_score

def extract_priors(dataset, args):
    start_time = time.time()

    # Load dataset cameras
    # We only need SH degree 0 just to initialize the scene loader
    gaussians = GaussianModel(0)
    scene = Scene(dataset, gaussians)
    
    train_cameras = scene.getTrainCameras()
    print(f"Loaded {len(train_cameras)} training cameras.")
    
    # Create reflection_prior folder
    save_dir = os.path.join(dataset.source_path, "reflection_prior")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")

    progress_bar = tqdm(train_cameras, desc="Extracting Priors (Tan-Ikeuchi)")

    for cam in progress_bar:
        ref_image_name = cam.image_name
        
        # original_image is [3, H, W] in [0, 1]
        img_tensor = cam.original_image.permute(1, 2, 0)
        img01 = img_tensor.cpu().numpy()
        
        # Calculate Ref Score using Tan-Ikeuchi
        final_score = tan_ikeuchi_score(img01, thresh=args.ti_thresh, bright_floor=args.ti_bright)
        
        # Convert to 8-bit image
        score_img = (final_score * 255).astype(np.uint8)
        
        # Save to disk
        save_path = os.path.join(save_dir, f"{ref_image_name}_ref_score.png")
        imageio.imwrite(save_path, score_img)

    end_time = time.time()
    print(f"Prior extraction complete in {end_time - start_time:.2f} seconds!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Extract Reflection Prior using Tan-Ikeuchi")
    
    lp = ModelParams(parser)
    
    # Add Tan-Ikeuchi specific parameters
    parser.add_argument("--ti_thresh", type=float, default=0.35, help="Tan-Ikeuchi intensity threshold")
    parser.add_argument("--ti_bright", type=float, default=0.6, help="Tan-Ikeuchi bright floor threshold")

    args = parser.parse_args()

    safe_state(False)

    extract_priors(lp.extract(args), args)
