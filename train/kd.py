"""Knowledge distillation from the locally trained ResNet-18 teacher."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import MEAN, STD
from train_teacher import resnet18_cifar


class Teacher(nn.Module):
    """Frozen teacher; `normalize_input=True` when the student sees raw [0,1] images."""

    def __init__(self, ckpt, dataset, normalize_input):
        super().__init__()
        nc = 10 if dataset == "cifar10" else 100
        self.net = resnet18_cifar(nc)
        self.net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.net.eval().requires_grad_(False)
        self.normalize_input = normalize_input
        self.register_buffer("mean", torch.tensor(MEAN[dataset]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD[dataset]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        if self.normalize_input:
            x = (x - self.mean) / self.std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.net(x).float()


class KDLoss(nn.Module):

    def __init__(self, base, alpha=0.7, T=4.0):
        super().__init__()
        self.base, self.alpha, self.T = base, alpha, T

    def forward(self, s_logits, target, t_logits):
        kd = F.kl_div(F.log_softmax(s_logits.float() / self.T, 1), F.softmax(t_logits / self.T, 1),
                      reduction="batchmean") * self.T**2
        return self.alpha * kd + (1 - self.alpha) * self.base(s_logits, target)
