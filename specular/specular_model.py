# specular/specular_model.py
# ============================================================
# Specular Model (ASG)
# First-class specular head for Specular-aware FastGS
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecularModel(nn.Module):
    """
    Specular Appearance Gaussian (ASG)

    This model predicts *true specular RGB* given:
    - per-Gaussian latent specular code
    - view direction
    - surface normal

    IMPORTANT:
    - This is NOT a post-hoc residual model.
    - Output is added directly to diffuse SH.
    """

    def __init__(
        self,
        spec_feat_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()

        self.spec_feat_dim = spec_feat_dim

        # Input:
        #   spec_feat      : F
        #   viewdir        : 3
        #   normal         : 3
        #   -------------------
        #   total          : F + 6
        input_dim = spec_feat_dim + 6

        layers = []
        dim = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            dim = hidden_dim

        # Final layer → RGB specular
        layers.append(nn.Linear(dim, 3))

        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    # ------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------
    def _init_weights(self):
        """
        Initialize weights conservatively so specular
        does not explode early in training.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(
        self,
        spec_feat: torch.Tensor,
        viewdir: torch.Tensor,
        normal: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            spec_feat : [N, F]   per-Gaussian latent specular code
            viewdir   : [N, 3]   view direction (world space, normalized)
            normal    : [N, 3]   surface normal (world space, normalized)

        Returns:
            specular_rgb : [N, 3]
        """

        # --------------------------------------------------------
        # Safety normalization (important for stability)
        # --------------------------------------------------------
        viewdir = F.normalize(viewdir, dim=1)
        normal = F.normalize(normal, dim=1)

        # --------------------------------------------------------
        # Concatenate inputs
        # --------------------------------------------------------
        h = torch.cat(
            [
                spec_feat,
                viewdir,
                normal
            ],
            dim=1
        )

        # --------------------------------------------------------
        # Predict specular RGB
        # --------------------------------------------------------
        specular_rgb = self.mlp(h)

        # --------------------------------------------------------
        # Optional: clamp or softplus
        # We keep it linear here; range will be shaped by loss.
        # --------------------------------------------------------

        return specular_rgb
