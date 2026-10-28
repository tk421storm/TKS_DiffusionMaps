"""Masked discrete diffusion (MaskGIT-style): training loss and the iterative sampler.

Training: hide a random fraction gamma(r) = cos(pi/2 * r), r ~ U(0,1), of the cells (replace them by the
mask token) and predict the originals. Loss = class-weighted cross-entropy over hidden cells whose
loss_mask is set (cells of buildings cut by the window edge never count).

Sampling: start with every free cell masked; each of `steps` rounds predicts all cells, samples a
category per masked cell, and keeps the most confident ones so that gamma((k+1)/steps) of the
originally masked cells remain hidden. `fixed` cells (inpainting / obstacles) are never masked.
"""
import math

import torch
import torch.nn.functional as F


def gamma(r):
    return torch.cos(r * math.pi / 2)


def masked_loss(model, x0, loss_mask, weights, mask_token, generator=None):
    B = x0.shape[0]
    r = torch.rand(B, device=x0.device, generator=generator)
    p = gamma(r).clamp(min=1.0 / x0[0].numel())
    hidden = torch.rand(x0.shape, device=x0.device, generator=generator) < p[:, None, None]
    xin = torch.where(hidden, torch.full_like(x0, mask_token), x0)
    logits = model(xin, hidden.float().mean(dim=(1, 2)))
    ce = F.cross_entropy(logits, x0, weight=weights, reduction="none")      # already weighted per target
    sel = (hidden & loss_mask).float()
    return (ce * sel).sum() / (weights[x0] * sel).sum().clamp(min=1.0)


@torch.no_grad()
def eval_loss(model, x0, loss_mask, weights, mask_token, ratios, seed):
    """Loss at fixed masked fractions with a fixed random pattern, so evaluations are comparable."""
    g = torch.Generator(device=x0.device).manual_seed(seed)
    out = {}
    for p in ratios:
        hidden = torch.rand(x0.shape, device=x0.device, generator=g) < p
        xin = torch.where(hidden, torch.full_like(x0, mask_token), x0)
        logits = model(xin, hidden.float().mean(dim=(1, 2)))
        ce = F.cross_entropy(logits, x0, weight=weights, reduction="none")
        sel = (hidden & loss_mask).float()
        acc = ((logits.argmax(1) == x0).float() * sel).sum() / sel.sum().clamp(min=1.0)
        out[p] = (((ce * sel).sum() / (weights[x0] * sel).sum().clamp(min=1.0)), acc)
    return out


@torch.no_grad()
def sample(model, n, size, num_classes, steps=16, temperature=1.0, conf_noise=4.5,
           fixed=None, generator=None, device="cuda", deterministic=False, logit_bias=None):
    """Returns (n, H, W) int64 category grids.

    fixed: optional (tokens (n,H,W), keep (n,H,W) bool) - cells where keep is True start (and stay) set.
    deterministic: argmax instead of sampling and no confidence noise (used for the ONNX comparison).
    logit_bias: optional (C,) added to the logits; -log(class weights) undoes the bias that
        class-weighted training puts on rare categories (logit adjustment).
    """
    H, W = size
    mask_token = num_classes
    x = torch.full((n, H, W), mask_token, dtype=torch.long, device=device)
    if fixed is not None:
        x = torch.where(fixed[1], fixed[0], x)
    start = (x == mask_token).flatten(1).sum(1)               # per-sample number of free cells
    for k in range(steps):
        masked = x == mask_token
        logits = model(x, masked.float().mean(dim=(1, 2))).permute(0, 2, 3, 1)
        if logit_bias is not None:
            logits = logits + logit_bias
        logp = torch.log_softmax(logits / max(temperature, 1e-6), dim=-1)
        if deterministic:
            conf, pick = logp.max(dim=-1)
        else:
            pick = torch.multinomial(logp.exp().reshape(-1, num_classes), 1, generator=generator).reshape(n, H, W)
            conf = torch.gather(logp, -1, pick[..., None])[..., 0]
            u = torch.rand(conf.shape, device=device, generator=generator).clamp(1e-20, 1.0)
            conf = conf + conf_noise * (1.0 - (k + 1) / steps) * -torch.log(-torch.log(u))   # Gumbel
        conf = torch.where(masked, conf, torch.full_like(conf, float("inf")))
        keep_hidden = torch.floor(math.cos((k + 1) / steps * math.pi / 2) * start.float()).long()
        flat = conf.flatten(1)
        kth = flat.sort(dim=1).values.gather(1, keep_hidden.clamp(max=flat.shape[1] - 1)[:, None])
        stay = (flat < kth).reshape(n, H, W) & (keep_hidden > 0)[:, None, None]
        x = torch.where(masked & ~stay, pick, x)
    return x
