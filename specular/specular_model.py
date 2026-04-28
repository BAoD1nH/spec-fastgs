# fastgs/specular/specular_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .spec_utils import RenderingEquationEncoding, ASGKernel, reflect

class SpecularModel(nn.Module):
    """
    Specular appearance as an additive term.
    Input: latent feat (from Gaussian), viewdir, normal
    Output: RGB specular [N,3]
    """
    def __init__(self, feat_dim=16, enc_levels=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.enc = RenderingEquationEncoding(3, enc_levels)
        enc_dim = 3 * (1 + 2 * enc_levels)
        self.color_head = nn.Sequential(
            nn.Linear(feat_dim + 2 * enc_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 3),
            nn.Tanh()
        )
        self.asg = ASGKernel(in_dim=3 + 3 + feat_dim)

    def forward(self, feat, viewdir, normal):
        # Normalize
        viewdir = F.normalize(viewdir, dim=-1)
        normal = F.normalize(normal, dim=-1)

        refl = reflect(viewdir, normal)
        e_view = self.enc(viewdir)
        e_refl = self.enc(refl)

        # ASG scalar lobe
        lobe = self.asg(refl, normal, feat)  # [N,1]

        rgb = self.color_head(torch.cat([feat, e_view, e_refl], dim=-1))
        return rgb * lobe

def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad_(False)

def train_setting(model, lr=2e-4):
    model.train()
    return torch.optim.Adam(model.parameters(), lr=lr)
