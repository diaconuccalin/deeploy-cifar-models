"""Train a CIFAR ResNet-18 teacher (for knowledge distillation) with GPU-side crop/flip.

Usage: python3.12 train_teacher.py cifar100 [--epochs 200]
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torchvision

from data import GPULoader, evaluate


def resnet18_cifar(nc):
    m = torchvision.models.resnet18(num_classes=nc)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["cifar10", "cifar100"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--out", default=os.path.expanduser("~/tallinn/runs"))
    a = p.parse_args()
    torch.manual_seed(0)
    nc = 10 if a.dataset == "cifar10" else 100
    out = os.path.join(a.out, f"teacher_resnet18_{a.dataset}")
    os.makedirs(out, exist_ok=True)
    tr, te = GPULoader(a.dataset, True, 128, normalize=True), GPULoader(a.dataset, False, 500, normalize=True)
    m = resnet18_cifar(nc).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(m.parameters(), lr=a.lr, momentum=0.9, nesterov=True, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tr), pct_start=0.05,
                                                div_factor=10, final_div_factor=1e3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, log = 0.0, []
    for ep in range(a.epochs):
        m.train()
        t0, tl = time.time(), 0.0
        for x, y in tr:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(m(x.contiguous(memory_format=torch.channels_last)), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item()
        acc = evaluate(m, te)
        if acc > best:
            best = acc
            torch.save(m.state_dict(), os.path.join(out, "best.pt"))
        log.append({"epoch": ep + 1, "loss": tl / len(tr), "acc": acc, "best": best, "time": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        json.dump(log, open(os.path.join(out, "log.json"), "w"))
    print(f"DONE best={best:.2f}", flush=True)
