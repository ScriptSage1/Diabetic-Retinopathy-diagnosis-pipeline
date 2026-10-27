"""Model definitions, checkpoint save/load, and probability helpers."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

NUM_GRADES = 5
REFERABLE_MIN_GRADE = 2


class SmallCNN(nn.Module):
    """Plain 4-block CNN trained from scratch: the 'no transfer learning' baseline (roadmap Module 4)."""

    def __init__(self, num_classes: int = NUM_GRADES, width: int = 32, drop: float = 0.3):
        super().__init__()
        chans = [3, width, width * 2, width * 4, width * 8]
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(chans[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(chans[-1], num_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(x))


def build_model(arch: str = "efficientnet_b0", num_classes: int = NUM_GRADES, pretrained: bool = True,
                drop_rate: float = 0.3, pretrained_file: str | None = None) -> nn.Module:
    """Create a classifier. `pretrained_file` loads local ImageNet weights (offline use); otherwise
    timm downloads them (needs internet). If pretrained=True and weights cannot be obtained this RAISES:
    we never silently fall back to random weights."""
    if arch == "small_cnn":
        return SmallCNN(num_classes)
    import timm

    if pretrained and pretrained_file:  # local weights (.pth / .safetensors); timm resets the classifier head
        return timm.create_model(arch, pretrained=True, num_classes=num_classes, drop_rate=drop_rate,
                                 pretrained_cfg_overlay=dict(file=str(pretrained_file)))
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional feature map (before global pooling)."""
    if isinstance(model, SmallCNN):
        return model.features[-3]  # BatchNorm of the last conv block (pre-ReLU, pre-pool)
    for name in ("bn2", "conv_head", "norm", "layer4"):
        if hasattr(model, name):
            return getattr(model, name)
    raise ValueError("no known last conv layer for this architecture")


# ----------------------------------------------------------------------------- probabilities
def grade_probs(logits: torch.Tensor | np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax over the 5 grades (optionally temperature-scaled)."""
    x = torch.as_tensor(logits, dtype=torch.float64) / float(temperature)
    return torch.softmax(x, dim=-1).numpy()


def referable_prob(probs: np.ndarray) -> np.ndarray:
    """P(referable) = P(grade 2) + P(grade 3) + P(grade 4), derived from the five-grade distribution."""
    return np.asarray(probs)[..., REFERABLE_MIN_GRADE:].sum(axis=-1)


# ----------------------------------------------------------------------------- checkpoints
def _versions() -> dict:
    v = {"torch": torch.__version__}
    try:
        import timm

        v["timm"] = timm.__version__
    except Exception:
        pass
    v["numpy"] = np.__version__
    return v


def atomic_save(obj, path: str | Path) -> None:
    """Write to a temp file then rename so an interrupted run never leaves a half-written checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, model, cfg: dict, epoch: int, history: list, best_metric: float,
                    optimizer=None, scheduler=None, scaler=None, extra: dict | None = None) -> None:
    ck = {
        "model_state": model.state_dict(),
        "cfg": cfg,
        "epoch": epoch,
        "history": history,
        "best_metric": best_metric,
        "versions": _versions(),
    }
    if optimizer is not None:
        ck["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        ck["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        ck["scaler_state"] = scaler.state_dict()
    if extra:
        ck.update(extra)
    atomic_save(ck, path)


def load_checkpoint(path: str | Path, device: str = "cpu", pretrained: bool = False):
    """Return (model in eval mode, checkpoint dict). Raises with a clear message if the file is unusable."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = ck["cfg"]
        model = build_model(cfg["arch"], NUM_GRADES, pretrained=pretrained, drop_rate=cfg.get("drop_rate", 0.3))
        model.load_state_dict(ck["model_state"])
    except Exception as e:
        raise RuntimeError(f"failed to load checkpoint {path}: {e}") from e
    return model.to(device).eval(), ck
