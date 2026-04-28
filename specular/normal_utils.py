# fastgs/specular/normal_utils.py

import torch
from utils.general_utils import build_rotation


@torch.no_grad()
def compute_gaussian_normals(gaussians):
    """
    Estimate per-Gaussian normal from ellipsoid geometry.

    Args:
        gaussians: GaussianModel

    Returns:
        normals: Tensor [N, 3], unit length
    """

    # scaling: [N, 3]
    scales = gaussians.get_scaling

    # rotation quaternion: [N, 4]
    rotations = gaussians._rotation

    # build rotation matrices: [N, 3, 3]
    R = build_rotation(rotations)

    # principal axis = axis with smallest scale
    # normal aligns with that axis direction
    min_axis = torch.argmin(scales, dim=1)  # [N]

    # pick axis direction
    normals = torch.zeros_like(gaussians.get_xyz)

    for i in range(3):
        mask = (min_axis == i)
        if mask.any():
            normals[mask] = R[mask, :, i]

    # normalize
    normals = normals / normals.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return normals