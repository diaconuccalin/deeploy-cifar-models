"""Dump a trained CCT state dict + its CIFAR test-set logits to .npz (for build_cct_onnx.py).

Usage: python3.12 export_cct_npz.py <run_dir> [--ckpt best.pt]
"""
import argparse
import os

import numpy as np
import torch

from data import GPULoader
from models import CCT

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--ckpt", default="best.pt")
a = p.parse_args()
nc = 100 if "cifar100" in a.run_dir else 10
m = CCT(nc)
m.load_state_dict(torch.load(os.path.join(a.run_dir, a.ckpt), map_location="cpu"))
m.eval()
dl = GPULoader("cifar100" if nc == 100 else "cifar10", False, 500, normalize=True, device="cpu")
with torch.no_grad():
    logits = torch.cat([m(x) for x, _ in dl]).numpy()
acc = 100.0 * (logits.argmax(1) == dl.y.numpy()).mean()
W = {k: v.numpy() for k, v in m.state_dict().items()}
W["_test_logits"] = logits
np.savez(os.path.join(a.run_dir, "weights.npz"), **W)
print(f"PyTorch test accuracy: {acc:.2f}% -> {os.path.join(a.run_dir, 'weights.npz')}")
