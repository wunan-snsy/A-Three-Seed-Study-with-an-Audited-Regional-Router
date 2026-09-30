"""Dataset loading and deterministic RGB-D corruptions for UCRR.

All transforms are defined in normalized depth coordinates. Synthetic corruptions
change depth/validity only; RGB and saliency GT remain in the reference frame.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


def read_manifest(path: str | Path, project_root: str | Path) -> list[dict[str, Any]]:
    root = Path(project_root)
    raw = Path(path).read_text(encoding="utf-8").splitlines()
    rows=[]
    for line in raw:
        if not line.strip():
            continue
        row=json.loads(line)
        if "paths" in row:
            for key in ("rgb", "depth", "gt"):
                row[key]=row["paths"][key]
        rows.append(row)
    for row in rows:
        for key in ("rgb", "depth", "gt"):
            row[key] = root / str(row[key]).replace("\\", "/")
    return rows


def _load_rgb(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_gray(path: Path, size: int, nearest: bool = False) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _stable_seed(*parts: str) -> int:
    msg = "|".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "little") % (2**32)


def corrupt_depth(depth: torch.Tensor, valid: torch.Tensor, family: str, severity: int,
                  rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a single reproducible synthetic corruption to 1xHxW tensors."""
    d, v = depth.clone(), valid.clone()
    _, h, w = d.shape
    if family == "clean":
        return d, v
    if family == "removal":
        return torch.zeros_like(d), torch.zeros_like(v)
    if family == "noise":
        sigma = (0.02, 0.05, 0.10)[severity - 1]
        noise = torch.from_numpy(rng.normal(0.0, sigma, size=tuple(d.shape)).astype(np.float32))
        return (d + noise * v).clamp(0, 1), v
    if family == "holes":
        frac = (0.10, 0.25, 0.40)[severity - 1]
        aspect = float(rng.uniform(0.5, 2.0))
        rh = min(h, max(1, round(math.sqrt(frac * h * w / aspect))))
        rw = min(w, max(1, round(frac * h * w / rh)))
        y, x = int(rng.integers(0, h - rh + 1)), int(rng.integers(0, w - rw + 1))
        v[:, y:y + rh, x:x + rw] = 0
        d[:, y:y + rh, x:x + rw] = 0
        return d, v
    if family == "blur":
        sigma = float((1, 2, 3)[severity - 1])
        radius = max(1, round(3 * sigma))
        # Blur depth*validity and validity separately; divide to avoid invalid zeros bleeding in.
        def blur(x: torch.Tensor) -> torch.Tensor:
            pil = Image.fromarray((x.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            out = pil.filter(ImageFilter.GaussianBlur(radius=radius))
            return torch.from_numpy(np.asarray(out, dtype=np.float32).copy() / 255.0).unsqueeze(0)
        den = blur(v).clamp_min(1e-5)
        return (blur(d * v) / den).clamp(0, 1), v
    if family == "shift":
        delta = (2, 5, 10)[severity - 1]
        dy, dx = ((-delta, 0), (-delta, delta), (0, delta), (delta, delta),
                  (delta, 0), (delta, -delta), (0, -delta), (-delta, -delta))[int(rng.integers(0, 8))]
        od, ov = torch.zeros_like(d), torch.zeros_like(v)
        sy0, sy1 = max(0, -dy), min(h, h - dy)
        sx0, sx1 = max(0, -dx), min(w, w - dx)
        dy0, dx0 = max(0, dy), max(0, dx)
        od[:, dy0:dy0 + sy1 - sy0, dx0:dx0 + sx1 - sx0] = d[:, sy0:sy1, sx0:sx1]
        ov[:, dy0:dy0 + sy1 - sy0, dx0:dx0 + sx1 - sx0] = v[:, sy0:sy1, sx0:sx1]
        return od, ov
    if family == "quantize":
        # Held-out corruption: quantization is never sampled by the training mixture.
        levels = (32, 16, 8)[severity - 1]
        return (torch.round(d * (levels - 1)) / (levels - 1)).clamp(0, 1), v
    if family == "noise_shift":
        # Held-out composition: constituents occur in training, but never jointly.
        noisy, noisy_valid = corrupt_depth(d, v, "noise", severity, rng)
        return corrupt_depth(noisy, noisy_valid, "shift", severity, rng)
    raise ValueError(f"Unknown corruption family: {family}")


class RGBDSaliencyDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], size: int = 320, training: bool = False,
                 corruption: str = "mixture", seed: int = 17):
        self.records, self.size, self.training = records, size, training
        self.corruption, self.seed = corruption, seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        r = self.records[index]
        rgb = _load_rgb(Path(r["rgb"]), self.size)
        depth = _load_gray(Path(r["depth"]), self.size)
        gt = (_load_gray(Path(r["gt"]), self.size, nearest=True) >= 0.5).float()
        valid = torch.ones_like(depth)
        py_rng = random.Random(self.seed + index)
        if self.training:
            if py_rng.random() < 0.5:
                rgb, depth, valid, gt = (x.flip(-1) for x in (rgb, depth, valid, gt))
        family = self.corruption
        if family == "mixture":
            family = py_rng.choice(("clean", "noise", "holes", "blur", "shift", "removal"))
        if family != "clean":
            sev = py_rng.randint(1, 3) if family in ("noise", "holes", "blur", "shift") else 0
            rng = np.random.default_rng(_stable_seed(self.seed.__str__(), str(r["dataset"]), str(r["id"]), family, str(sev), str(index)))
            depth, valid = corrupt_depth(depth, valid, family, sev, rng)
        rgb = (rgb - torch.tensor((0.485, 0.456, 0.406))[:, None, None]) / torch.tensor((0.229, 0.224, 0.225))[:, None, None]
        return {"rgb": rgb, "depth": depth, "valid": valid, "gt": gt,
                "dataset": r["dataset"], "id": r["id"], "partition": r.get("partition", "unknown")}
