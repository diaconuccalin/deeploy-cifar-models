"""Model definitions matching the Deeploy reference graphs.

- CCT: exactly the architecture of DeeployTest/Tests/Models/CCT/FP32/CCT_2_32_32_128
  (2 conv tokenizer layers, 2 transformer blocks, 2 heads, dim 128, MLP ratio 1, seq-pool).
- MiniMobileNetV2: the DeeployTest miniMobileNetv2 topology, widened by `width`, with a global
  average-pool head, standard MobileNetV2 activations (ReLU after stem/expand/dw, linear project).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------- CCT


class Attention(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.heads, self.hd).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.heads, self.hd).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))  # 0.125 in the ref graph
        a = a.softmax(-1)
        return self.proj((a @ v).transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    """x = norm1(x + attn(pre_norm(x))); x = x + linear2(gelu(linear1(x)))"""

    def __init__(self, dim, heads, mlp_dim, drop_path=0.0, dropout=0.0):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, dim)
        self.act = nn.GELU(approximate="tanh")  # Deeploy's FP32 GELU is the tanh approximation
        self.drop_path = drop_path
        self.dropout = nn.Dropout(dropout)

    def _dp(self, x):
        if not self.training or self.drop_path == 0.0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_path
        return x * keep / (1 - self.drop_path)

    def forward(self, x):
        x = x + self._dp(self.self_attn(self.pre_norm(x)))
        x = self.norm1(x)
        x = x + self._dp(self.linear2(self.dropout(self.act(self.linear1(x)))))
        return x


class CCT(nn.Module):

    def __init__(self, num_classes=10, dim=128, heads=2, layers=2, mlp_dim=128, img=32, drop_path=0.1,
                 dropout=0.0):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, dim, 3, 1, 1, bias=False), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        n_tok = (img // 4)**2
        self.positional_emb = nn.Parameter(torch.zeros(1, n_tok, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_dim, dpr[i], dropout) for i in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.attention_pool = nn.Linear(dim, 1)  # its bias cancels in the softmax; dropped at export
        self.fc = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.positional_emb, std=0.2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.tokenizer(x).flatten(2).transpose(1, 2) + self.positional_emb
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        w = self.attention_pool(x).transpose(1, 2).softmax(-1)  # B,1,N
        return self.fc((w @ x).squeeze(1))


# ------------------------------------------------------------------------------ MiniMobileNetV2


class ConvBN(nn.Module):

    def __init__(self, cin, cout, k, s, groups=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = relu

    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.relu(x) if self.relu else x


class InvRes(nn.Module):

    def __init__(self, cin, cout, s, expand=2):
        super().__init__()
        h = cin * expand
        self.expand = ConvBN(cin, h, 1, 1)
        self.dw = ConvBN(h, h, 3, s, groups=h)
        self.project = ConvBN(h, cout, 1, 1, relu=False)
        self.residual = s == 1 and cin == cout

    def forward(self, x):
        y = self.project(self.dw(self.expand(x)))
        return x + y if self.residual else y


class MiniMobileNetV2(nn.Module):
    """Reference topology x width: stem 16w (s2) | IR(16w->16w, s2) | IR(16w->16w, s1, residual) |
    IR(16w->32w, s1) | global avg pool | FC."""

    def __init__(self, num_classes=10, width=4):
        super().__init__()
        c, e = 16 * width, 32 * width
        self.stem = ConvBN(3, c, 3, 2)
        self.blocks = nn.Sequential(InvRes(c, c, 2), InvRes(c, c, 1), InvRes(c, e, 1))
        self.fc = nn.Linear(e, num_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.fc(x.mean((2, 3)))
