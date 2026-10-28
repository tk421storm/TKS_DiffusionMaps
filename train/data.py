"""Window set loading for training: everything goes to the device once, augmentation happens there.

Keeping the whole set on the GPU matters under WSL: per-step host->device copies hung the GPU bridge
(vmbus_teardown_gpadl) on Windows 10. v1 is ~6k windows x 48x48, about 14 MB plus masks.
"""
import json
import os

import numpy as np
import torch


class WindowSet:
    def __init__(self, set_dir, split, device):
        d = np.load(os.path.join(set_dir, split + ".npz"))
        self.manifest = json.load(open(os.path.join(set_dir, "manifest.json"), encoding="utf-8"))
        cats = self.manifest["config"]["categories"]
        self.num_classes = max(c["id"] for c in cats) + 1
        self.mask_token = self.num_classes            # extra token id for "masked / unknown"
        self.names = {c["id"]: c["name"] for c in cats}
        self.chars = {c["id"]: c.get("char", "?") for c in cats}
        self.structure_ids = [c["id"] for c in cats if c.get("structure")]
        self.cells_np = d["cells"]
        self.loss_mask_np = d["loss_mask"]
        self.cells = torch.from_numpy(d["cells"].astype(np.int64)).to(device)
        self.loss_mask = torch.from_numpy(d["loss_mask"]).to(device)
        self.device = device

    def __len__(self):
        return len(self.cells)

    def class_weights(self, power, cap):
        """Per-category loss weights ~ frequency^-power over cells that count for the loss, capped, mean 1 per cell."""
        counts = np.bincount(self.cells_np[self.loss_mask_np].ravel(), minlength=self.num_classes).astype(np.float64)
        freq = counts / counts.sum()
        w = np.where(freq > 0, np.maximum(freq, 1e-12) ** -power, 0.0)
        w = w / (freq * w).sum()                      # expected weight of a random cell = 1
        w = np.minimum(w, cap)
        return torch.tensor(w, dtype=torch.float32, device=self.device), freq

    def batches(self, batch, generator, augment=True):
        """One epoch of shuffled batches; each batch gets one random rotation/flip (8 symmetries)."""
        perm = torch.randperm(len(self), device=self.device, generator=generator)
        for i in range(0, len(self) - batch + 1, batch):
            idx = perm[i:i + batch]
            x, m = self.cells[idx], self.loss_mask[idx]
            if augment:
                k = int(torch.randint(0, 8, (1,), generator=_cpu_gen(generator)).item())
                x, m = dihedral(x, k), dihedral(m, k)
            yield x, m


_cpu_gens = {}


def _cpu_gen(gpu_gen):
    # The symmetry choice is drawn on the CPU so it needs no device->host sync per batch.
    g = _cpu_gens.get(id(gpu_gen))
    if g is None:
        g = torch.Generator().manual_seed(gpu_gen.initial_seed())
        _cpu_gens[id(gpu_gen)] = g
    return g


def dihedral(x, k):
    """k in 0..7: rotate by 90*(k%4) degrees, then mirror if k>=4. Works on (B, H, W)."""
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if k >= 4 else x
