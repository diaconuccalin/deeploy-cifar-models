"""Train MiniMobileNetV2 (width 4, global-avg-pool head) on CIFAR: float phase, then QAT.

Inputs are pixel/255 (no mean/std normalization), matching the deployed uint8 input.
Usage: python3.12 train_mmnv2.py cifar10|cifar100 [--epochs 200] [--qat-epochs 40]
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from data import GPULoader, evaluate, loaders
from models import MiniMobileNetV2
from quant import QMiniMobileNetV2

p = argparse.ArgumentParser()
p.add_argument("dataset", choices=["cifar10", "cifar100"])
p.add_argument("--width", type=int, default=4)
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--qat-epochs", type=int, default=40)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--qat-lr", type=float, default=0.005)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--workers", type=int, default=5)
p.add_argument("--autoaugment", action="store_true")
p.add_argument("--skip-float", action="store_true", help="reuse float_best.pt")
p.add_argument("--tag", default="", help="suffix for the run directory")
p.add_argument("--label-smoothing", type=float, default=0.0)
p.add_argument("--teacher", default="", help="teacher checkpoint for knowledge distillation")
p.add_argument("--kd-alpha", type=float, default=0.7)
p.add_argument("--kd-T", type=float, default=4.0)
p.add_argument("--out", default=os.path.expanduser("~/tallinn/runs"))
a = p.parse_args()

torch.manual_seed(0)
nc = 10 if a.dataset == "cifar10" else 100
out = os.path.join(a.out, f"mmnv2_w{a.width}_{a.dataset}{a.tag}")
os.makedirs(out, exist_ok=True)
dev = "cuda"
if a.autoaugment:
    train_dl, test_dl = loaders(a.dataset, a.batch, a.workers, normalize=False, autoaugment=True)
else:  # crop+flip only: keep everything on the GPU
    train_dl, test_dl = GPULoader(a.dataset, True, a.batch), GPULoader(a.dataset, False, 500)
crit = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
teacher = None
if a.teacher:
    from kd import KDLoss, Teacher
    teacher = Teacher(a.teacher, a.dataset, normalize_input=True).to(dev)
    kd_crit = KDLoss(crit, a.kd_alpha, a.kd_T)
log = []


def run(model, epochs, lr, tag, params):
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(train_dl), pct_start=0.05,
                                                anneal_strategy="cos", div_factor=10, final_div_factor=1e3)
    best = 0.0
    for ep in range(epochs):
        model.train()
        if tag == "qat":  # BN running stats stay frozen (they are folded into the weights)
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        t0, tl = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            out_s = model(x)
            loss = kd_crit(out_s, y, teacher(x)) if teacher is not None else crit(out_s, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item()
        acc = evaluate(model, test_dl)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(out, f"{tag}_best.pt"))
        torch.save(model.state_dict(), os.path.join(out, f"{tag}_last.pt"))
        log.append({"phase": tag, "epoch": ep + 1, "loss": tl / len(train_dl), "acc": acc, "best": best,
                    "time": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        json.dump(log, open(os.path.join(out, "log.json"), "w"))
    return best


def param_groups(model, wd):
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    rest = [p for n, p in model.named_parameters() if p.ndim <= 1]
    return [{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0}]


fm = MiniMobileNetV2(nc, a.width).to(dev)
print(f"params: {sum(p.numel() for p in fm.parameters())}", flush=True)
if a.skip_float:
    fm.load_state_dict(torch.load(os.path.join(out, "float_best.pt")))
else:
    run(fm, a.epochs, a.lr, "float", param_groups(fm, a.wd))
    fm.load_state_dict(torch.load(os.path.join(out, "float_best.pt")))
print(f"float acc: {evaluate(fm, test_dl):.2f}", flush=True)

qm = QMiniMobileNetV2(fm).to(dev)
qm.train()
with torch.no_grad():  # initialize activation scales on one batch
    for m in qm.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    qm(next(iter(train_dl))[0].to(dev))
print(f"QAT init acc: {evaluate(qm, test_dl):.2f}", flush=True)
torch.save(qm.state_dict(), os.path.join(out, "qat_init.pt"))  # quantized model before any QAT
best = run(qm, a.qat_epochs, a.qat_lr, "qat", param_groups(qm, a.wd / 10)) if a.qat_epochs > 0 else 0.0
print(f"DONE qat best={best:.2f}", flush=True)
