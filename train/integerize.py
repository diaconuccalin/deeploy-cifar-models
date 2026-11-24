"""Fold a QAT MiniMobileNetV2 into integer weights + RequantShift parameters (numpy .npz).

RequantShift semantics (Deeploy): y = clip(floor((x*mul + add + div/2) / div)), div = 2^D.
For each layer D is the largest value (<= 30) such that |x*mul + add| < 2^31 is guaranteed
for every possible input (worst-case accumulator bound), so the int32 arithmetic never overflows.

Usage: python3.12 integerize.py <run_dir> [--ckpt qat_best.pt]
"""
import argparse
import os

import numpy as np
import torch

from data import GPULoader
from models import MiniMobileNetV2
from quant import QMiniMobileNetV2

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--ckpt", default="qat_best.pt")
a = p.parse_args()

nc = 100 if "cifar100" in a.run_dir else 10
width = int(a.run_dir.split("_w")[1].split("_")[0])
qm = QMiniMobileNetV2(MiniMobileNetV2(nc, width))
for m in qm.modules():  # mark scales as initialized so loading doesn't re-init
    if hasattr(m, "initialized"):
        m.initialized = True
qm.load_state_dict(torch.load(os.path.join(a.run_dir, a.ckpt), map_location="cpu"))
qm.eval().double()

INT32 = 2**31 - 1
P = {}


def choose_D(acc_bound, M, B):
    """Largest D <= 30 with acc_bound*|mul| + |add| + 2^(D-1) < 2^31 for all channels."""
    for D in range(30, 0, -1):
        mul = np.round(M * 2**D)
        add = np.round(B * 2**D)
        if np.all(acc_bound * np.abs(mul) + np.abs(add) + 2**(D - 1) < INT32) and np.all(np.abs(mul) < INT32):
            return D, mul, add
    raise ValueError("no valid shift")


def conv_layer(name, L, s_x, x_max):
    w, b = L.folded()
    sw = L.weight_scale(w)
    w_int = torch.clamp(torch.round(w / sw.view(-1, 1, 1, 1)), -127, 127)
    s_y = L.act.s
    M = (sw * s_x / s_y).numpy()
    B = (b / s_y).numpy()
    acc_bound = w_int.abs().sum((1, 2, 3)).numpy() * x_max
    D, mul, add = choose_D(acc_bound, M, B)
    P[f"{name}.weight"] = w_int.numpy()
    P[f"{name}.mul"], P[f"{name}.add"], P[f"{name}.div"] = mul, add, np.array(2.0**D)
    P[f"{name}.signed"] = np.array(0.0 if L.relu else 1.0)
    return s_y.item(), (255 if L.relu else 128)


def scalar_rqs(name, M, x_max):
    D, mul, add = choose_D(np.array([x_max]), np.array([M]), np.array([0.0]))
    P[f"{name}.mul"], P[f"{name}.add"], P[f"{name}.div"] = mul[0], add[0], np.array(2.0**D)


with torch.no_grad():
    s, xm = 1.0 / 255, 255
    s, xm = conv_layer("stem", qm.stem, s, xm)
    for i, blk in enumerate([qm.b1, qm.b2, qm.b3], 1):
        s_in = s
        s, xm = conv_layer(f"b{i}.expand", blk.expand, s, xm)
        s, xm = conv_layer(f"b{i}.dw", blk.dw, s, xm)
        s, xm = conv_layer(f"b{i}.project", blk.project, s, xm)
        if blk.residual:
            sm = max(s_in, s)
            scalar_rqs(f"b{i}.skip", s_in / sm, 128)  # RQS on the block input
            scalar_rqs(f"b{i}.branch", s / sm, 128)  # RQS on the project output
            s_add = blk.add_act.s.item()
            scalar_rqs(f"b{i}.add", sm / s_add, 256)
            s = s_add
    # global average pool: DW conv of ones over HxW + RQS with mul/div = 1/(HW), same scale
    P["gap.scale"] = np.array(s)
    # classifier
    s_fc = qm.fc_weight_scale()
    P["fc.weight"] = torch.clamp(torch.round(qm.fc.weight / s_fc), -127, 127).numpy()
    P["fc.bias"] = torch.round(qm.fc.bias / (s_fc * s)).numpy()
    # logit scale: chosen on the training set so that the int8 logits (round + clip to [-128,127])
    # agree best with the float argmax (avoids ties from a scale dominated by rare large logits)
    dl = GPULoader("cifar100" if nc == 100 else "cifar10", True, 1000, device="cpu", augment=False)
    logits = torch.cat([qm(x.double()) for x, _ in dl])
    ref = logits.argmax(1)
    best = None
    for q in [1.0, 0.9999, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]:
        s_try = torch.quantile(logits.abs().amax(1), q).item() / 127
        li = torch.clamp(torch.round(logits / s_try), -128, 127)
        # argmax picks the first maximum, as on the device
        agree = (li.argmax(1) == ref).float().mean().item()
        print(f"  logit scale q={q}: agreement {100 * agree:.2f}%")
        if best is None or agree > best[0]:
            best = (agree, s_try)
    s_log = best[1]
    acc_bound = np.abs(P["fc.weight"]).sum(1).max() * 128 + np.abs(P["fc.bias"]).max()
    scalar_rqs("fc", (s_fc * s / s_log).item(), acc_bound)
    P["meta.width"], P["meta.classes"] = np.array(width), np.array(nc)

out = os.path.join(a.run_dir, "int_params.npz")
np.savez(out, **P)
print("saved", out)
for k in sorted(P):
    if k.endswith(".div"):
        print(f"  {k:16s} D={int(np.log2(P[k]))}")
