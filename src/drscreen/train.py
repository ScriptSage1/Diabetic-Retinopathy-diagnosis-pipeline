"""Training loop: AMP, class-weighted CE, warmup+cosine, early stopping, per-epoch checkpoints, resume, time budget."""
from __future__ import annotations

import csv
import dataclasses
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import DRDataset
from .metrics import fit_temperature, grade_metrics, referable_labels, referable_prob
from .models import build_model, grade_probs, load_checkpoint, save_checkpoint


@dataclass
class TrainConfig:
    arch: str = "efficientnet_b0"
    img_size: int = 384
    epochs: int = 30
    batch_size: int = 24
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_epochs: float = 1.0
    drop_rate: float = 0.3
    label_smoothing: float = 0.0
    class_weight_power: float = 0.5  # weight_k ∝ (1/freq_k)^power, normalised; 0 = unweighted
    patience: int = 8
    monitor: str = "qwk"  # validation metric used for best checkpoint / early stopping: qwk | auc_ref | loss
    amp: bool = True
    num_workers: int = 4
    seed: int = 42
    pretrained: bool = True
    pretrained_file: str | None = None
    use_ben_graham: bool = True
    max_minutes: float | None = None  # wall-clock budget for the whole call; stops cleanly between epochs
    resume: bool = True
    out_dir: str = "artifacts"
    run_name: str = "efficientnet_b0"
    max_train_batches: int | None = None  # smoke tests only
    robust_aug: bool = False  # add blur / JPEG / illumination augmentation to TRAINING images only
    init_from: str | None = None  # start from the weights of this checkpoint (fine-tuning); ignored when resuming a run

    def to_dict(self):
        return dataclasses.asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(labels: np.ndarray, power: float, n_classes: int = 5) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    w = (1.0 / np.maximum(counts, 1.0)) ** power
    return torch.tensor(w / w.mean() if power else np.ones(n_classes), dtype=torch.float32)


def make_loader(cache, labels, indices, cfg: TrainConfig, train: bool, device: str) -> DataLoader:
    ds = DRDataset(cache, labels, indices, train=train, use_ben_graham=cfg.use_ben_graham, seed=cfg.seed, robust=bool(cfg.robust_aug) and train)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, drop_last=train and len(ds) > cfg.batch_size, num_workers=cfg.num_workers,
                      pin_memory=(device == "cuda"), generator=g if train else None, persistent_workers=False)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: str, amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic inference over a loader (no augmentation). Returns (logits [N,5], labels [N])."""
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp and device == "cuda"):
            o = model(x)
        outs.append(o.float().cpu())
        ys.append(y)
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()


def _val_metrics(logits, y):
    probs = grade_probs(logits)
    yb = referable_labels(y)
    pr = referable_prob(probs)
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(yb, pr)) if len(np.unique(yb)) == 2 else float("nan")
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    loss = float(-logp[np.arange(len(y)), y].mean())
    g = grade_metrics(y, probs.argmax(1))
    return {"val_loss": loss, "val_qwk": g["qwk"], "val_auc_ref": auc, "val_acc": g["accuracy_5"]}


def _score(h: dict, monitor: str) -> float:
    v = {"qwk": h["val_qwk"], "auc_ref": h["val_auc_ref"], "loss": -h["val_loss"]}[monitor]
    return -1e9 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def train(cfg: TrainConfig, cache: np.ndarray, df: pd.DataFrame, device: str | None = None, log=print) -> dict:
    """Train on rows with split=='train', validate on split=='val'. df needs columns cache_idx, grade, split.

    Writes <out_dir>/<run_name>/{last.pt, best.pt, history.csv, config.json}. last.pt holds optimizer, scheduler and
    scaler state so an interrupted run resumes at the next epoch. Returns a summary dict.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    labels = np.zeros(len(cache), int)
    labels[df["cache_idx"].to_numpy()] = df["grade"].to_numpy()
    tr_idx = df.loc[df["split"] == "train", "cache_idx"].to_numpy()
    va_idx = df.loc[df["split"] == "val", "cache_idx"].to_numpy()
    if cfg.max_train_batches:  # smoke test
        tr_idx = tr_idx[: cfg.max_train_batches * cfg.batch_size]
    train_loader = make_loader(cache, labels, tr_idx, cfg, True, device)
    val_loader = make_loader(cache, labels, va_idx, cfg, False, device)

    model = build_model(cfg.arch, 5, cfg.pretrained, cfg.drop_rate, cfg.pretrained_file).to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if cfg.init_from and not (run_dir / "last.pt").is_file():
        model.load_state_dict(torch.load(cfg.init_from, map_location=device, weights_only=False)["model_state"])
        log(f"[init] weights loaded from {cfg.init_from}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total, warm = cfg.epochs * steps_per_epoch, int(cfg.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))) * 0.99 + 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")
    crit = nn.CrossEntropyLoss(weight=class_weights(labels[tr_idx], cfg.class_weight_power).to(device), label_smoothing=cfg.label_smoothing)

    history, best, bad, start_epoch = [], -1e9, 0, 1
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if cfg.resume and last_path.is_file():
        try:
            ck = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state"])
            opt.load_state_dict(ck["optimizer_state"])
            sched.load_state_dict(ck["scheduler_state"])
            scaler.load_state_dict(ck["scaler_state"])
            history, best, bad, start_epoch = ck["history"], ck["best_metric"], ck.get("bad_epochs", 0), ck["epoch"] + 1
            log(f"[resume] continuing from epoch {start_epoch} (best {best:.4f})")
        except Exception as e:  # corrupt/incompatible last.pt: start fresh, but say so
            log(f"[resume] could not resume from {last_path}: {e}; starting from scratch")
            history, best, bad, start_epoch = [], -1e9, 0, 1

    t_start, epoch_times, stop_reason = time.time(), [], "completed all epochs"
    for epoch in range(start_epoch, cfg.epochs + 1):
        if cfg.max_minutes and epoch_times:
            if (time.time() - t_start) + float(np.mean(epoch_times)) * 1.15 > cfg.max_minutes * 60:
                stop_reason = f"stopped: wall-clock budget of {cfg.max_minutes} min would be exceeded"
                break
        t0 = time.time()
        model.train()
        run_loss, n_seen = 0.0, 0
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if device == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=cfg.amp and device == "cuda"):
                loss = crit(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch} batch {bi}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            n_seen += len(y)
        logits, yv = predict_logits(model, val_loader, device, cfg.amp)
        h = {"epoch": epoch, "train_loss": run_loss / max(1, n_seen), **_val_metrics(logits, yv),
             "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(h)
        epoch_times.append(h["seconds"])
        score = _score(h, cfg.monitor)
        improved = score > best
        if improved:
            best, bad = score, 0
        else:
            bad += 1
        log(f"epoch {epoch:02d}/{cfg.epochs} train_loss {h['train_loss']:.4f} val_loss {h['val_loss']:.4f} "
            f"val_qwk {h['val_qwk']:.4f} val_auc_ref {h['val_auc_ref']:.4f} ({h['seconds']:.0f}s){' *best*' if improved else ''}")
        save_checkpoint(last_path, model, cfg.to_dict(), epoch, history, best, opt, sched, scaler, extra={"bad_epochs": bad})
        if improved:
            save_checkpoint(best_path, model, cfg.to_dict(), epoch, history, best, extra={"best_epoch": epoch})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if bad >= cfg.patience:
            stop_reason = f"early stopping: no improvement in validation {cfg.monitor} for {cfg.patience} epochs"
            break
    (run_dir / "config.json").write_text(__import__("json").dumps(cfg.to_dict(), indent=2))
    return {"run_dir": str(run_dir), "epochs_completed": len(history), "best_epoch": int(max(history, key=lambda r: _score(r, cfg.monitor))["epoch"]) if history else 0,
            "best_monitor": best, "stop_reason": stop_reason, "device": device,
            "train_seconds": float(sum(h["seconds"] for h in history)), "best_path": str(best_path), "last_path": str(last_path)}


def verify_checkpoint_roundtrip(ckpt_path, cache, labels, indices, cfg: TrainConfig, device: str = "cpu", atol: float = 1e-5) -> dict:
    """Reload the checkpoint twice and confirm identical logits on the same inputs (reproducible inference)."""
    loader = make_loader(cache, labels, indices, dataclasses.replace(cfg, num_workers=0), False, device)
    m1, _ = load_checkpoint(ckpt_path, device)
    m2, _ = load_checkpoint(ckpt_path, device)
    l1, _ = predict_logits(m1, loader, device, amp=False)
    l2, _ = predict_logits(m2, loader, device, amp=False)
    return {"max_abs_diff": float(np.abs(l1 - l2).max()), "identical_within_tol": bool(np.allclose(l1, l2, atol=atol)), "finite": bool(np.isfinite(l1).all())}
