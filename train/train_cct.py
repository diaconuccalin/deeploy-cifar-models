"""Train FP32 CCT_2_32_32_128 on CIFAR-10/100 (CCT paper recipe, scaled to this model size).

Usage: python3.12 train_cct.py cifar10|cifar100 [--epochs 300]
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torchvision.transforms import v2

from data import evaluate, loaders
from models import CCT

p = argparse.ArgumentParser()
p.add_argument("dataset", choices=["cifar10", "cifar100"])
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--lr", type=float, default=55e-5)
p.add_argument("--wd", type=float, default=6e-2)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--workers", type=int, default=6)
p.add_argument("--out", default=os.path.expanduser("~/tallinn/runs"))
p.add_argument("--tag", default="")
p.add_argument("--teacher", default="")
p.add_argument("--kd-alpha", type=float, default=0.7)
p.add_argument("--kd-T", type=float, default=4.0)
a = p.parse_args()

torch.manual_seed(0)
nc = 10 if a.dataset == "cifar10" else 100
out = os.path.join(a.out, f"cct_{a.dataset}{a.tag}")
os.makedirs(out, exist_ok=True)
dev = "cuda"

train_dl, test_dl = loaders(a.dataset, a.batch, a.workers, normalize=True, autoaugment=True)
model = CCT(nc).to(dev)
print(f"params: {sum(p.numel() for p in model.parameters())}", flush=True)

decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "positional_emb" not in n]
no_decay = [p for n, p in model.named_parameters() if not (p.ndim > 1 and "positional_emb" not in n)]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0}],
                        lr=a.lr)
steps_per_epoch = len(train_dl)
total, warm = a.epochs * steps_per_epoch, a.warmup * steps_per_epoch
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: s / warm if s < warm else 1e-5 / a.lr + (1 - 1e-5 / a.lr) * 0.5 *
    (1 + math.cos(math.pi * (s - warm) / (total - warm))))
mix = v2.RandomChoice([v2.MixUp(alpha=0.8, num_classes=nc), v2.CutMix(alpha=1.0, num_classes=nc)])
crit = nn.CrossEntropyLoss(label_smoothing=0.1)
teacher = None
if a.teacher:
    from kd import KDLoss, Teacher
    teacher = Teacher(a.teacher, a.dataset, normalize_input=False).to(dev)  # CCT input is normalized
    kd_crit = KDLoss(crit, a.kd_alpha, a.kd_T)

best, log = 0.0, []
for ep in range(a.epochs):
    model.train()
    t0, tot_loss = time.time(), 0.0
    for x, y in train_dl:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        x, y = mix(x, y)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_s = model(x)
            loss = kd_crit(out_s, y, teacher(x)) if teacher is not None else crit(out_s, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tot_loss += loss.item()
    acc = evaluate(model, test_dl)
    if acc > best:
        best = acc
        torch.save(model.state_dict(), os.path.join(out, "best.pt"))
    torch.save(model.state_dict(), os.path.join(out, "last.pt"))
    log.append({"epoch": ep + 1, "loss": tot_loss / steps_per_epoch, "acc": acc, "best": best,
                "time": time.time() - t0})
    print(json.dumps(log[-1]), flush=True)
    json.dump(log, open(os.path.join(out, "log.json"), "w"))
print(f"DONE best={best:.2f} last={acc:.2f}", flush=True)
