# ============================================================
# Gaussian Model (Spec-Gaussian + FastGS compatible)
# ============================================================

import torch
import numpy as np
from torch import nn
import os

from utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func,
    build_rotation, get_linear_noise_func,
    strip_symmetric, build_scaling_rotation,
    flip_align_view, get_minimum_axis
)

from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud


class GaussianModel:
    def __init__(self, sh_degree: int, asg_dim: int = 24):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            cov = L @ L.transpose(1, 2)
            return strip_symmetric(cov)

        # SH config
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        # ✅ ASG config
        self.asg_dim = asg_dim

        # gaussian params
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._features_asg = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self.optimizer = None

        # activations
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    # ------------------------------------------------------------
    # GETTERS
    # ------------------------------------------------------------

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_asg_features(self):
        return self._features_asg

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # ------------------------------------------------------------
    # NORMAL (SG requirement)
    # ------------------------------------------------------------

    def get_normal_axis(self, dir_pp_normalized=None):
        normal_axis = self.get_minimum_axis
        normal_axis, _ = flip_align_view(normal_axis, dir_pp_normalized)
        normal = normal_axis / normal_axis.norm(dim=1, keepdim=True)
        return normal

    @property
    def get_minimum_axis(self):
        return get_minimum_axis(self.get_scaling, self.get_rotation)

    # ------------------------------------------------------------
    # INIT FROM PCD
    # ------------------------------------------------------------

    def create_from_pcd(self, pcd: BasicPointCloud):

        xyz = torch.tensor(np.asarray(pcd.points)).float().cuda()
        colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()

        sh_color = RGB2SH(colors)

        features = torch.zeros(
            (colors.shape[0], 3, (self.max_sh_degree + 1) ** 2),
            device="cuda"
        )
        features[:, :3, 0] = sh_color

        # ✅ ASG feature (SG)
        asg_features = torch.zeros((colors.shape[0], self.asg_dim), device="cuda")

        dist2 = torch.clamp_min(distCUDA2(xyz), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        rots = torch.zeros((xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((xyz.shape[0], 1), device="cuda")
        )

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous())
        self._features_asg = nn.Parameter(asg_features.requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    # ------------------------------------------------------------
    # TRAINING SETUP
    # ------------------------------------------------------------

    def training_setup(self, training_args):

        self.percent_dense = training_args.percent_dense

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        param_groups = [
            {"params": [self._xyz], "lr": training_args.position_lr_init, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20, "name": "f_rest"},
            {"params": [self._features_asg], "lr": training_args.feature_lr, "name": "f_asg"},  # ✅ SG core
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    # ------------------------------------------------------------
    # DENSIFY / PRUNE (FROM FASTGS)
    # ------------------------------------------------------------

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
        new_asg
    ):
        self._xyz = torch.cat([self._xyz, new_xyz], dim=0)
        self._features_dc = torch.cat([self._features_dc, new_features_dc], dim=0)
        self._features_rest = torch.cat([self._features_rest, new_features_rest], dim=0)
        self._features_asg = torch.cat([self._features_asg, new_asg], dim=0)

        self._opacity = torch.cat([self._opacity, new_opacity], dim=0)
        self._scaling = torch.cat([self._scaling, new_scaling], dim=0)
        self._rotation = torch.cat([self._rotation, new_rotation], dim=0)

    # (giữ nguyên densify_and_prune từ FastGS nếu cần)


