#!/usr/bin/env python
"""
Pre-launch gate: confirm all four models instantiate and forward a single batch at n_in=10.

Only resnet is verified at 10 channels so far (it trained the all_channels run), so the real risk is
the three transformer / UPerNet wrappers whose patch-embed adapters must rebuild for n_in != 3. This
catches any ConvNeXt / Swin / SegFormer 10-channel adaptation bug NOW (it would otherwise throw on the
first forward pass mid-matrix). CPU-only, one tiny forward each; no training, no GPU run.
"""
import os
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve()
SRC_PKG = HERE.parents[1]          # src/ML_sdfi_fastai2  (for `import train`, `import utils.utils`)
SRC = HERE.parents[2]              # src                  (for `import ML_sdfi_fastai2.*`)
for p in (str(SRC_PKG), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
torch.set_grad_enabled(False)

import train  # noqa: E402  (the trainer module holds the wrapper classes)

N_IN, N_CLASS, HW = 10, 11, 256
x = torch.randn(1, N_IN, HW, HW)
results = []


def run(name, build):
    try:
        model = build().eval()
        logits = model(x)
        ok = tuple(logits.shape) == (1, N_CLASS, HW, HW)
        results.append((name, "PASS" if ok else "BAD SHAPE", tuple(logits.shape)))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))


run("SegFormer-b1 (segformer-b1)",
    lambda: train.SegFormerWrapper("nvidia/segformer-b1-finetuned-ade-512-512",
                                   num_classes=N_CLASS, n_in=N_IN, pretrained=True))
run("ConvNeXt-base+UPerNet (convnext_base_upernet)",
    lambda: train.ConvNeXtUPerNetWrapper("convnext_base", num_classes=N_CLASS, n_in=N_IN,
                                         pretrained=True))
run("Swin-base+UPerNet (swin-base-upernet)",
    lambda: train.SwinUPerNetWrapper("openmmlab/upernet-swin-base", num_classes=N_CLASS, n_in=N_IN,
                                     pretrained=True))

# resnet34 + UNet: already confirmed at 10ch by the completed all_channels training run. Try a
# standalone DynamicUnet build too (best-effort; the trained run is the authoritative proof).
def build_resnet():
    from torchvision.models import resnet34
    from fastai.vision.learner import create_unet_model
    return create_unet_model(resnet34, N_CLASS, (HW, HW), pretrained=True, n_in=N_IN)
run("resnet34+UNet (resnet34) [also confirmed by all_channels training run]", build_resnet)

print("\n=== n_in=10 single-forward check (1x10x256x256 -> expect 1x11x256x256) ===")
worst_ok = True
for name, status, detail in results:
    print(f"  [{status:9}] {name}\n              {detail}")
    if status != "PASS":
        worst_ok = False
print("\nALL PASS" if worst_ok else "\nSOME CHECKS FAILED (see above)")
sys.exit(0 if worst_ok else 1)
