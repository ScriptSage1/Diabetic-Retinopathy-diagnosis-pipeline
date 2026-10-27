"""Consistent matplotlib figures for the notebook, app and report. Every function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from .metrics import GRADE_NAMES, binary_metrics, referable_labels, referable_prob, reliability_bins

C_MAIN, C_ALT, C_GREY = "#1f5f8b", "#c8553d", "#8a8a8a"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})


def _save(fig, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_binary_confusion(m: dict, path, title="Referable DR screening: confusion matrix") -> str:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cm, cmap="Blues")
    lab = [["TN\ncorrectly passed", "FP\nfalse alarm"], ["FN\nMISSED referable", "TP\ncorrectly flagged"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{lab[i][j]}", ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Predicted\nnon-referable", "Predicted\nreferable"])
    ax.set_yticks([0, 1], ["Actual\nnon-referable", "Actual\nreferable (positive)"])
    ax.set_title(f"{title}\nthreshold = {m['threshold']:.3f}, n = {m['n']}", fontsize=10)
    ax.spines[:].set_visible(False)
    return _save(fig, path)


def plot_grade_confusion(cm, path, title="Five-grade confusion matrix") -> str:
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, data, fmt, sub in ((axs[0], cm, "d", "counts"), (axs[1], norm, ".2f", "row-normalised (recall per true grade)")):
        ax.imshow(data, cmap="Blues")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=8,
                        color="white" if data[i, j] > data.max() / 2 else "black")
        ax.set_xticks(range(5), [str(i) for i in range(5)])
        ax.set_yticks(range(5), [str(i) for i in range(5)])
        ax.set_xlabel("Predicted grade")
        ax.set_ylabel("True grade")
        ax.set_title(sub, fontsize=9)
        ax.spines[:].set_visible(False)
    fig.suptitle(title, fontweight="bold")
    return _save(fig, path)


def plot_roc_pr(y_grade, probs5, threshold: float, path, label="model") -> str:
    y, p = referable_labels(y_grade), referable_prob(probs5)
    m = binary_metrics(y, p, threshold)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2))
    fpr, tpr, _ = roc_curve(y, p)
    axs[0].plot(fpr, tpr, color=C_MAIN, label=f"{label} (AUC = {m['roc_auc']:.3f})")
    axs[0].plot([0, 1], [0, 1], "--", color=C_GREY, label="chance")
    axs[0].plot(1 - m["specificity"], m["sensitivity"], "o", color=C_ALT, label=f"operating point (t = {threshold:.2f})")
    axs[0].set(xlabel="1 - specificity (false positive rate)", ylabel="Sensitivity (true positive rate)", title="ROC curve")
    axs[0].legend(loc="lower right", fontsize=8)
    pr, rc, _ = precision_recall_curve(y, p)
    axs[1].plot(rc, pr, color=C_MAIN, label=f"{label} (AP = {m['pr_auc']:.3f})")
    axs[1].axhline(m["prevalence"], ls="--", color=C_GREY, label=f"prevalence = {m['prevalence']:.2f}")
    axs[1].plot(m["sensitivity"], m["precision"], "o", color=C_ALT, label="operating point")
    axs[1].set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title="Precision-recall curve", ylim=(0, 1.02))
    axs[1].legend(loc="lower left", fontsize=8)
    return _save(fig, path)


def plot_reliability(y_grade, probs_before, probs_after, path, n_bins: int = 10) -> str:
    y = referable_labels(y_grade)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, probs, name in ((axs[0], probs_before, "before temperature scaling"), (axs[1], probs_after, "after temperature scaling")):
        rb = reliability_bins(referable_prob(probs), y, n_bins)
        mp, fp, ct = np.array(rb["mean_pred"], float), np.array(rb["frac_pos"], float), np.array(rb["count"])
        ok = ct > 0
        ax.plot([0, 1], [0, 1], "--", color=C_GREY, label="perfect calibration")
        ax.plot(mp[ok], fp[ok], "o-", color=C_MAIN, label="observed")
        ax2 = ax.twinx()
        ax2.bar((np.arange(n_bins) + 0.5) / n_bins, ct, width=0.9 / n_bins, color=C_GREY, alpha=0.25)
        ax2.set_yticks([])
        ax2.spines[:].set_visible(False)
        ax.set(xlabel="Predicted P(referable)", title=name, xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="lower right", fontsize=8)
    axs[0].set_ylabel("Observed fraction referable")
    fig.suptitle("Reliability diagram (grey bars: samples per bin)", fontweight="bold")
    return _save(fig, path)


def plot_learning_curves(history, path, title="Training history") -> str:
    ep = [h["epoch"] for h in history]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axs[0].plot(ep, [h["train_loss"] for h in history], label="train", color=C_MAIN)
    axs[0].plot(ep, [h["val_loss"] for h in history], label="validation", color=C_ALT)
    axs[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axs[1].plot(ep, [h["val_qwk"] for h in history], color=C_MAIN)
    axs[1].set(xlabel="Epoch", ylabel="Validation QWK", title="Quadratic weighted kappa")
    axs[2].plot(ep, [h["val_auc_ref"] for h in history], color=C_MAIN)
    axs[2].set(xlabel="Epoch", ylabel="Validation ROC-AUC (referable)", title="Referable AUC")
    axs[0].legend()
    fig.suptitle(title, fontweight="bold")
    return _save(fig, path)


def plot_threshold_sweep(y_grade, probs5, chosen: float, path) -> str:
    y, p = referable_labels(y_grade), referable_prob(probs5)
    ts = np.linspace(0.01, 0.99, 99)
    ms = [binary_metrics(y, p, t) for t in ts]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(ts, [m["sensitivity"] for m in ms], label="sensitivity", color=C_ALT)
    ax.plot(ts, [m["specificity"] for m in ms], label="specificity", color=C_MAIN)
    ax.plot(ts, [m["precision"] for m in ms], label="precision (PPV)", color=C_GREY, ls=":")
    ax.axvline(chosen, color="k", ls="--", lw=0.8, label=f"chosen threshold {chosen:.2f}")
    ax.set(xlabel="Decision threshold on P(referable)", ylabel="Metric", title="Sensitivity / specificity trade-off", ylim=(0, 1.02))
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_class_distribution(counts_by_split: dict, path) -> str:
    """counts_by_split: {'train': [n0..n4], 'val': [...], 'test': [...]}"""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.8 / len(counts_by_split)
    for i, (split, c) in enumerate(counts_by_split.items()):
        x = np.arange(5) + i * w - 0.4 + w / 2
        bars = ax.bar(x, c, w, label=f"{split} (n={sum(c)})")
        for b, v in zip(bars, c):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(5), GRADE_NAMES, rotation=15, ha="right")
    ax.set(ylabel="Images", title="Class distribution per split (stratified on grade)")
    ax.legend()
    return _save(fig, path)


def plot_image_grid(images, titles, path, ncols: int = 5, suptitle: str | None = None, figw: float = 2.4) -> str:
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    lines = max([str(t).count(chr(10)) + 1 for t in titles] + [1])
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * figw, nrows * (figw + 0.32 * lines + 0.1)))
    axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")
    for ax, im, t in zip(axs.flat, images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=8, pad=4)
    if suptitle:
        fig.suptitle(suptitle, fontweight="bold", fontsize=10)
    fig.tight_layout(h_pad=1.6)
    if suptitle:
        fig.subplots_adjust(top=1 - (0.55 / (nrows * (figw + 0.32 * lines + 0.1))) - 0.04)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_quality_degradation(deg_rows: list, clean: dict, path) -> str:
    """Share of images flagged (reject or enhance) per synthetic degradation and severity; dashed line = flag rate on unmodified images."""
    kinds = list(dict.fromkeys(r["kind"] for r in deg_rows))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for si, (sev, col) in enumerate(((1, "#9ecae1"), (2, "#4292c6"), (3, "#08519c"))):
        vals = [next(r for r in deg_rows if r["kind"] == k and r["severity"] == sev) for k in kinds]
        ax.bar(np.arange(len(kinds)) + si * 0.27 - 0.27, [v["reject"] + v["enhance"] for v in vals], 0.27, color=col, label=f"severity {sev}")
    base = 1 - clean["good"] / max(1, clean["n"])
    ax.axhline(base, color=C_ALT, ls="--", label=f"flagged among unmodified images ({base:.0%})")
    ax.set_xticks(range(len(kinds)), kinds, rotation=20)
    ax.set(ylabel="Share flagged (reject or enhance)", ylim=(0, 1.32), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
           title=f"Quality gate response to synthetic degradations (n={deg_rows[0]['n']} images each)")
    ax.legend(fontsize=8, ncol=2, loc="upper center")
    return _save(fig, path)


def plot_bars(labels, values, path, title, ylabel, errs=None, ylim=(0, 1.05), colors=None) -> str:
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(labels)), 4.2))
    err = None if errs is None else np.array(errs).T
    ax.bar(range(len(labels)), values, yerr=err, color=colors or C_MAIN, capsize=3)
    for i, v in enumerate(values):
        ax.text(i, min(v + 0.02, ylim[1] - 0.05), f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=0, ha="center", fontsize=8)
    ax.set(title=title, ylabel=ylabel, ylim=ylim)
    return _save(fig, path)


def plot_robustness(names, std_rows, rob_rows, path) -> str:
    """Grouped bars (standard vs robustness-augmented model) of ungated sensitivity and specificity per image condition."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    x = np.arange(len(names))
    for ax, key, title in ((axs[0], "ungated_sensitivity", "Sensitivity (referable cases flagged)"), (axs[1], "ungated_specificity", "Specificity (non-referable cases passed)")):
        a = [np.nan if r[key] is None else r[key] for r in std_rows]
        b = [np.nan if r[key] is None else r[key] for r in rob_rows]
        ax.bar(x - 0.2, a, 0.4, color=C_MAIN, label="standard training")
        ax.bar(x + 0.2, b, 0.4, color=C_ALT, label="robustness-augmented training")
        ax.set_xticks(x, names, rotation=30, ha="right")
        ax.set(title=title, ylim=(0, 1.1))
    axs[0].set_ylabel("Metric at the validation-selected threshold")
    h, l = axs[0].get_legend_handles_labels()
    fig.suptitle("Effect of robustness augmentation on degraded images (same grade-stratified test subset, no quality gate)", fontweight="bold", fontsize=10)
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)
