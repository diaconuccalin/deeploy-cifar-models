"""Quantization-aware MiniMobileNetV2 that integerizes exactly to the Deeploy graph
(Conv -> RequantShift chains, residual RQS+Add+RQS, DW-conv global avg pool, Gemm -> RQS).

Scales: input = 1/255 (raw uint8 pixels). Every ConvBN has BN folded into the weights (running
stats frozen), per-output-channel int8 weights (symmetric, 127 levels) and an LSQ-learned
output scale; ReLU layers output uint8 [0,255], linear layers int8 [-128,127].
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ConvBN, InvRes, MiniMobileNetV2


class _RoundSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def _grad_scale(x, g):
    return (x - x * g).detach() + x * g


def fq(x, s, lo, hi):
    """Fake-quantize x with scale s to integers in [lo, hi] (LSQ-style, STE on rounding)."""
    return torch.clamp(_RoundSTE.apply(x / s), lo, hi) * s


class ActQ(nn.Module):
    """Learned output scale (LSQ)."""

    def __init__(self, unsigned):
        super().__init__()
        self.lo, self.hi = (0, 255) if unsigned else (-128, 127)
        self.log_s = nn.Parameter(torch.zeros(()))
        self.initialized = False

    @property
    def s(self):
        return self.log_s.exp()

    def init_from(self, x):
        # percentile init: cover 99.99% of |x|
        v = x.detach().abs().flatten()
        k = max(1, int(0.9999 * v.numel()))
        self.log_s.data = torch.log(v.kthvalue(k).values.clamp_min(1e-4) / self.hi)
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.init_from(x)
        s = _grad_scale(self.s, 1.0 / math.sqrt(x.numel() / x.shape[0] * self.hi))
        return fq(x, s, self.lo, self.hi)


class QConvBN(nn.Module):

    def __init__(self, m: ConvBN):
        super().__init__()
        self.conv, self.bn, self.relu = m.conv, m.bn, m.relu
        self.act = ActQ(unsigned=m.relu)

    def folded(self):
        bn = self.bn
        inv = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = self.conv.weight * inv.view(-1, 1, 1, 1)
        b = bn.bias - bn.running_mean * inv
        return w, b

    def weight_scale(self, w):
        return w.detach().abs().amax((1, 2, 3)).clamp_min(1e-8) / 127.0

    def forward(self, x):
        w, b = self.folded()
        sw = self.weight_scale(w)
        wq = fq(w, sw.view(-1, 1, 1, 1), -127, 127)
        c = self.conv
        y = F.conv2d(x, wq, b, c.stride, c.padding, c.dilation, c.groups)
        return self.act(y)  # clamp at 0 implements the ReLU for unsigned layers


class QInvRes(nn.Module):

    def __init__(self, m: InvRes):
        super().__init__()
        self.expand, self.dw, self.project = QConvBN(m.expand), QConvBN(m.dw), QConvBN(m.project)
        self.residual = m.residual
        if self.residual:
            self.add_act = ActQ(unsigned=False)

    def mid_scale(self, x_scale):
        return torch.maximum(x_scale, self.project.act.s).detach()

    def forward(self, x, x_scale=None):
        y = self.project(self.dw(self.expand(x)))
        if not self.residual:
            return y
        sm = self.mid_scale(x_scale)
        return self.add_act(fq(x, sm, -128, 127) + fq(y, sm, -128, 127))


class QMiniMobileNetV2(nn.Module):

    def __init__(self, m: MiniMobileNetV2):
        super().__init__()
        self.stem = QConvBN(m.stem)
        self.b1, self.b2, self.b3 = (QInvRes(b) for b in m.blocks)
        self.fc = m.fc
        self.spatial = None

    def fc_weight_scale(self):
        return self.fc.weight.detach().abs().max().clamp_min(1e-8) / 127.0

    def features(self, x):
        x = fq(x, 1.0 / 255, 0, 255)  # input is pixel/255: exact
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x, self.b1.project.act.s)
        x = self.b3(x)
        s = self.b3.project.act.s.detach()
        return fq(x.mean((2, 3)), s, -128, 127)  # DW-conv(ones)+RQS(1/HW) average pool

    def forward(self, x):
        g = self.features(x)
        wq = fq(self.fc.weight, self.fc_weight_scale(), -127, 127)
        return F.linear(g, wq, self.fc.bias)  # float logits for the loss; int8 RQS added at export
