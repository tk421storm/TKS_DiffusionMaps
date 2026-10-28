"""U-Net denoiser for masked discrete diffusion over category grids.

Input: tokens (B, H, W) int64, values 0..C-1 or C (= masked), and the masked fraction (B,) float.
Output: logits (B, C, H, W) for the original category of every cell.
Kept to ONNX-friendly ops (Gather, Conv, GroupNorm, MatMul, Softmax, Resize) for in-game inference.
H and W must be divisible by 2**(levels-1).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ratio_embedding(ratio, dim):
    """Sinusoidal embedding of the masked fraction (0..1)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=ratio.device, dtype=torch.float32) / half)
    a = ratio[:, None].float() * 1000.0 * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, 2 * cout)            # FiLM: scale and shift from the masked fraction
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.t(t)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        d = C // self.heads
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, d, H * W).unbind(1)
        attn = torch.softmax((q.transpose(-1, -2) @ k) * d ** -0.5, dim=-1)   # (B, heads, HW, HW)
        out = (v @ attn.transpose(-1, -2)).reshape(B, C, H, W)
        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(self, num_classes, base=64, mults=(1, 2, 4), nres=2, attn_levels=(2,), heads=4, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        tdim = 4 * base
        self.tdim = tdim
        self.t_mlp = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.embed = nn.Embedding(num_classes + 1, base)   # +1: the mask token
        self.stem = nn.Conv2d(base, base, 3, padding=1)
        chs = [base * m for m in mults]

        # Encoder; remember channel counts of every skip.
        self.down = nn.ModuleList()
        skip_chs, prev = [base], base
        for lvl, ch in enumerate(chs):
            for _ in range(nres):
                self.down.append(nn.ModuleList([ResBlock(prev, ch, tdim, dropout),
                                                Attention(ch, heads) if lvl in attn_levels else nn.Identity()]))
                prev = ch
                skip_chs.append(ch)
            if lvl < len(chs) - 1:
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skip_chs.append(ch)

        self.mid = nn.ModuleList([ResBlock(prev, prev, tdim, dropout), Attention(prev, heads),
                                  ResBlock(prev, prev, tdim, dropout)])

        # Decoder mirrors the encoder, consuming one skip per block.
        self.up = nn.ModuleList()
        for lvl in reversed(range(len(chs))):
            ch = chs[lvl]
            for _ in range(nres + 1):
                self.up.append(nn.ModuleList([ResBlock(prev + skip_chs.pop(), ch, tdim, dropout),
                                              Attention(ch, heads) if lvl in attn_levels else nn.Identity()]))
                prev = ch
            if lvl > 0:
                self.up.append(nn.Conv2d(ch, ch, 3, padding=1))   # applied after nearest upsampling
        self.out = nn.Sequential(nn.GroupNorm(8, prev), nn.SiLU(), nn.Conv2d(prev, num_classes, 1))
        self.base = base

    def forward(self, tokens, ratio):
        t = self.t_mlp(ratio_embedding(ratio, self.base))
        h = self.stem(self.embed(tokens).permute(0, 3, 1, 2))
        skips = [h]
        for m in self.down:
            if isinstance(m, nn.ModuleList):
                h = m[1](m[0](h, t))
            else:
                h = m(h)
            skips.append(h)
        h = self.mid[0](h, t)
        h = self.mid[1](h)
        h = self.mid[2](h, t)
        for m in self.up:
            if isinstance(m, nn.ModuleList):
                h = m[1](m[0](torch.cat([h, skips.pop()], dim=1), t))
            else:
                h = m(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return self.out(h)


def build(cfg, num_classes):
    return UNet(num_classes, base=cfg["base"], mults=tuple(cfg["mults"]), nres=cfg["nres"],
                attn_levels=tuple(cfg["attn_levels"]), heads=cfg["heads"], dropout=cfg["dropout"])
