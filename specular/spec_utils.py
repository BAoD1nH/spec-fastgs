# fastgs/specular/spec_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def reflect(v, n):
    # v, n: [N,3], assume normalized
    return v - 2.0 * (v * n).sum(dim=-1, keepdim=True) * n

class RenderingEquationEncoding(nn.Module):
    """Simple positional encoding for view/reflection dirs"""
    def __init__(self, dims=3, levels=4):
        super().__init__()
        self.levels = levels
        self.freqs = 2.0 ** torch.arange(levels)

    def forward(self, x):
        # x: [N,3]
        outs = [x]
        for f in self.freqs.to(x.device):
            outs += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(outs, dim=-1)

class ASGKernel(nn.Module):
    """
    Anisotropic Spherical Gaussian kernel:
      lobe_dir from normal, sharpness learned from latent
    """
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, 2),   # [amplitude, sharpness]
            nn.Softplus()
        )

    def forward(self, refl_dir, normal, feat):
        # refl_dir, normal: [N,3]; feat: [N,F]
        x = torch.cat([refl_dir, normal, feat], dim=-1)
        amp_sharp = self.mlp(x)
        amp = amp_sharp[:, :1]
        sharp = amp_sharp[:, 1:2] + 1e-6
        # cosine lobe around normal
        cos_theta = torch.clamp((refl_dir * normal).sum(dim=-1, keepdim=True), min=0.0)
        return amp * torch.exp(sharp * (cos_theta - 1.0))