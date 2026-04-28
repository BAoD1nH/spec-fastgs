# fastgs/train_specular.py
import torch, random, os
from argparse import ArgumentParser
from scene import Scene
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_fastgs
from specular import SpecularModel
from specular.specular_model import train_setting, freeze_module
from specular.specular_detector import detect_specular_gaussians

def freeze_gaussians(g):
    for t in [g._xyz, g._scaling, g._rotation, g._opacity, g._features_dc, g._features_rest]:
        t.requires_grad_(False)

def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipe = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--spec_iters", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--feat_dim", type=int, default=16)
    args = get_combined_args(parser)

    mult = 0.5  # same as render.py

    # Load FastGS model
    gaussians = GaussianModel(model.sh_degree, optimizer_type="default")
    scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=True)
    freeze_gaussians(gaussians)

    # Background color
    bg = torch.tensor([0,0,0], device="cuda", dtype=torch.float32)

    # Detect specular Gaussians
    specular_mask = detect_specular_gaussians(scene, gaussians, pipe.extract(args), bg, mult).cuda()
    if specular_mask.sum() == 0:
        print("No specular Gaussians detected. Exiting.")
        return

    print(f"Detected {specular_mask.float().mean().item()*100:.2f}% Gaussians as Specular.")

    # Init specular
    spec = SpecularModel(feat_dim=args.feat_dim).cuda()
    optim = train_setting(spec, lr=args.lr)

    cams = scene.getTrainCameras()
    for it in range(args.spec_iters):
        cam = random.choice(cams)
        out = render_fastgs(
            cam, gaussians, pipe.extract(args), bg, mult,
            specular_model=spec,
            specular_mask=specular_mask
        )
        pred = out["render"]
        gt = cam.original_image[:3]

        loss = torch.mean(torch.abs(pred - gt))
        optim.zero_grad()
        loss.backward()
        optim.step()

        if it % 100 == 0:
            print(f"[Specular] iter {it} | L1 {loss.item():.4f}")

    # Save
    sp_dir = os.path.join(args.model_path, "specular")
    os.makedirs(sp_dir, exist_ok=True)
    torch.save(spec.state_dict(), os.path.join(sp_dir, "specular.pth"))
    print("Saved specular checkpoint.")

if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()