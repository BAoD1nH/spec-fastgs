#!/usr/bin/env python3
"""Fail-fast verification for a Spec-FastGS training environment."""

from __future__ import print_function

import importlib
import shutil
import subprocess
import sys


REQUIRED_MODULES = (
    ("numpy", "NumPy"),
    ("PIL", "Pillow"),
    ("imageio", "ImageIO"),
    ("plyfile", "plyfile"),
    ("tqdm", "tqdm"),
    ("websockets", "websockets"),
    ("gdown", "gdown"),
    ("simple_knn._C", "simple-knn CUDA extension"),
    ("diff_gaussian_rasterization_fastgs", "FastGS CUDA rasterizer"),
    ("fused_ssim", "fused-ssim CUDA extension"),
)


def module_version(module):
    return getattr(module, "__version__", "installed")


def main():
    failures = []
    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)

    try:
        import torch
    except Exception as error:
        print("[FAIL] PyTorch: {}".format(error))
        return 1

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)

    for module_name, label in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            print("[ OK ] {} ({})".format(label, module_version(module)))
        except Exception as error:
            failures.append(label)
            print("[FAIL] {}: {}".format(label, error))

    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            output = subprocess.check_output(
                [nvcc, "--version"], stderr=subprocess.STDOUT
            ).decode("utf-8", errors="replace")
            version_line = output.strip().splitlines()[-1]
            print("nvcc:", version_line)
        except Exception as error:
            print("[WARN] Could not query nvcc: {}".format(error))
    else:
        print("[WARN] nvcc not found; rebuilding CUDA extensions will not work.")

    if not torch.cuda.is_available():
        failures.append("CUDA device")
        print("[FAIL] torch.cuda.is_available() is False")
    else:
        try:
            tensor = torch.tensor([1.0], device="cuda")
            result = (tensor * 2).item()
            print("[ OK ] GPU: {} (CUDA tensor result: {})".format(
                torch.cuda.get_device_name(0), result
            ))
        except Exception as error:
            failures.append("CUDA tensor operation")
            print("[FAIL] CUDA tensor operation: {}".format(error))

    if failures:
        print("\nEnvironment verification FAILED: {}".format(", ".join(failures)))
        return 1

    print("\nEnvironment verification PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
