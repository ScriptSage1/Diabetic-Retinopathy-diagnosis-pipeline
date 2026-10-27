"""Figures for the learning guide (report/guide/figures). Two kinds, kept apart on purpose:

* explanatory diagrams drawn from scratch (anatomy schematics, flowcharts, illustrative metric examples); every illustrative
  number is labelled as such in the figure and in its caption;
* figures computed from the real project: the project's own preprocessing / augmentation code applied to a real APTOS image,
  the trained classifier's first-layer feature maps, the real EfficientNet-B0 layer shapes, the real learning-rate schedule,
  bootstrap resampling of the saved test predictions, PPV/NPV derived from the measured sensitivity and specificity, the
  lesion locator's output, expert-annotated e-ophtha lesion crops.

    python scripts/make_guide_figures.py        (CPU, about a minute; needs the local example images and artifacts)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "report" / "guide" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
EX = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "drscreen" / "aptos_examples"
EOPHTHA = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "drscreen" / "eophtha"

INK, INK2, RULE = "#1c2220", "#4a524e", "#cfcabd"
ACC, ACC_S = "#1f5f55", "#e3edea"
REF, REF_S = "#a4412c", "#f5e6e1"
AMB, AMB_S = "#8a6212", "#f4ead3"
IND, IND_S = "#34507a", "#e9eef6"
SLA, SLA_S = "#4e5963", "#eef0f2"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.06, "figure.facecolor": "white"})

GRADE_EXAMPLES = {0: "02358b47ea89", 1: "111898ab463d", 2: "026dcd9af143", 3: "070f67572d03", 4: "0981195eb9fb"}
GRADE_NAMES = ["0 · No apparent DR", "1 · Mild NPDR", "2 · Moderate NPDR", "3 · Severe NPDR", "4 · Proliferative DR"]


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("saved", name)


def read_rgb(p):
    from drscreen.preprocess import read_image

    return read_image(p)


def canvas_ax(fig=None, size=(7, 3.2)):
    fig = fig or plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * size[1] / size[0])
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc=ACC_S, ec=ACC, tc=INK, fs=8.2, bold=False, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.25,rounding_size={r}", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", linespacing=1.25, wrap=True)


def arrow(ax, p, q, color=INK2, lw=1.1, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


# =============================================================================== Part I: biology
def fig_eye_anatomy():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(-2.25, 2.35)
    ax.set_ylim(-1.62, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, fc="#fbfaf6", ec="#9a958a", lw=2.4))           # sclera
    ax.add_patch(Wedge((0, 0), 0.97, -118, 118, width=0.06, fc="#e9a99a", ec="none"))  # retina lines the back of the eye
    ax.add_patch(Wedge((0, 0), 1.0, -118, 118, width=0.03, fc="#b06a52", ec="none", alpha=0.6))  # choroid
    ax.add_patch(Arc((-0.78, 0), 0.62, 1.15, theta1=110, theta2=250, color="#5d7f9a", lw=2.6))  # cornea bulge
    ax.add_patch(Rectangle((-0.83, 0.18), 0.05, 0.45, fc="#4a6b55", ec="none"))     # iris (upper)
    ax.add_patch(Rectangle((-0.83, -0.63), 0.05, 0.45, fc="#4a6b55", ec="none"))    # iris (lower)
    ax.add_patch(Ellipse((-0.62, 0), 0.24, 0.62, fc="#e8eef4", ec="#5d7f9a", lw=1.4))  # lens
    ax.add_patch(Polygon([[0.92, -0.20], [1.55, -0.30], [1.55, -0.04], [0.95, -0.04]], closed=True, fc="#f2dcc7", ec="#9a7a5a", lw=1.2))  # optic nerve
    ax.add_patch(Circle((0.955, -0.12), 0.05, fc="#f6d36b", ec="#a2852d", lw=1))    # optic disc
    ax.add_patch(Circle((0.985, 0.17), 0.035, fc="#7a3b2e", ec="none"))              # fovea
    for yo in (-0.30, 0.0, 0.30):   # light rays focused onto the fovea
        ax.plot([-2.1, -0.75], [yo, yo], color=AMB, lw=1.1, alpha=0.9)
        ax.plot([-0.75, 0.985], [yo, 0.17], color=AMB, lw=1.1, alpha=0.9)
    ax.annotate("", xy=(-1.55, 0.0), xytext=(-2.1, 0.0), arrowprops=dict(arrowstyle="-|>", color=AMB, lw=1.2))
    ax.text(-2.1, 0.40, "light", color=AMB, fontsize=9, fontweight="bold")
    lab = [("Cornea\n(clear front window;\ndoes most of the focusing)", (-1.08, -0.38), (-1.75, -0.85)),
           ("Iris and pupil\n(the iris is the coloured ring;\nthe pupil is the opening)", (-0.805, 0.58), (-1.05, 1.32)),
           ("Lens\n(fine focusing)", (-0.60, 0.25), (-0.15, 1.28)),
           ("Vitreous humour\n(clear gel filling the eye)", (-0.05, -0.35), (-0.75, -1.22)),
           ("Retina\n(light-sensitive tissue\nlining the back)", (0.62, 0.72), (1.12, 1.25)),
           ("Macula / fovea\n(sharp central vision)", (0.99, 0.18), (1.30, 0.62)),
           ("Optic disc\n(where the nerve and\nvessels leave the eye)", (0.96, -0.14), (1.28, -0.88)),
           ("Optic nerve\n(to the brain)", (1.45, -0.18), (1.75, 0.15)),
           ("Sclera\n(tough white wall)", (0.30, -0.95), (0.55, -1.25))]
    for t, xy, xt in lab:
        ax.annotate(t, xy=xy, xytext=xt, fontsize=7.6, color=INK, ha="left" if xt[0] > xy[0] else "right" if xt[0] < xy[0] - 0.2 else "center", va="center",
                    arrowprops=dict(arrowstyle="-", color="#8b877d", lw=0.7, shrinkA=2, shrinkB=1))
    ax.text(-2.2, -1.6, "Schematic cross-section seen from above (not to scale). Front of the eye on the left.", fontsize=7, color=INK2)
    save(fig, "eye_anatomy.png")


def fig_retina_layers():
    fig, ax = canvas_ax(size=(7.2, 3.3))
    H = 100 * 3.3 / 7.2
    ax.text(2, H - 3, "Vitreous (inside of the eye)", fontsize=8, color=INK2, va="top")
    layers = [("Nerve fibre layer (retinal blood vessels in red):\naxons running to the optic disc", 34, "#e8eef4"), ("Ganglion cells (output neurons of the retina)", 28, "#dfe7f0"),
              ("Bipolar and other connecting cells", 21, "#eef0f2"), ("Photoreceptors: rods and cones", 11, "#f7efdc"), ("Pigment epithelium", 7, "#e5d2c4"),
              ("Choroid (vascular layer behind the retina)", 2.5, "#f1d4cc")]
    tops = [38, 31, 24, 16, 9, 5.5]
    for (name, y, col), top in zip(layers, tops):
        ax.add_patch(Rectangle((2, y), 62, top - y, fc=col, ec="white", lw=1.2))
        ax.text(66, (y + top) / 2, name, fontsize=7.6, va="center", color=INK)
    for i in range(14):        # rods and cones
        x = 4 + i * 4.3
        if i % 3 == 1:
            ax.add_patch(Polygon([[x, 11.3], [x + 2.2, 11.3], [x + 1.1, 15.4]], closed=True, fc="#c58b2a", ec="none"))
        else:
            ax.add_patch(Rectangle((x + 0.6, 11.3), 1.0, 4.4, fc="#8a6212", ec="none"))
    for x in (12, 41):          # retinal vessels in the inner retina
        ax.add_patch(Circle((x, 33.5), 1.6, fc=REF, ec="none"))
    for x in (22, 46):          # light passes through the transparent inner layers to the photoreceptors
        arrow(ax, (x, H - 6), (x, 16.2), color=AMB, lw=1.4)
    ax.text(23, 39.2, "light", fontsize=7.5, color=AMB, fontweight="bold")
    for x in (30, 55):          # signals travel back towards the vitreous side
        arrow(ax, (x, 13), (x, 35), color=ACC, lw=1.4)
    ax.text(31, 18.5, "signal", fontsize=7.5, color=ACC, fontweight="bold")
    ax.text(2, 0, "Simplified, not to scale. Light crosses the transparent inner layers, is absorbed by rods (dim light) and cones (colour, fine detail),\n"
                  "and the resulting signals pass through bipolar cells to ganglion cells, whose axons form the optic nerve.", fontsize=6.6, color=INK2, va="top")
    ax.set_ylim(-5, H)
    save(fig, "retina_layers.png")


def fig_fundus_annotated():
    from drscreen.lesion import LesionAnalyzer

    img = read_rgb(EX / f"{GRADE_EXAMPLES[0]}.png")
    an = LesionAnalyzer(ROOT / "artifacts" / "lesion", device="cpu")
    out = an.analyze(img)
    base = out["canvas_plain"]
    od, fo = np.asarray(out["od_xy"], float), np.asarray(out["fovea_xy"], float)
    dd = float(np.linalg.norm(fo - od)) / 2.5  # about 2.5 disc diameters separate disc and fovea in the locator's prior
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1, 1]})
    axs[0].imshow(img)
    axs[0].set_title(f"As photographed ({img.shape[1]} × {img.shape[0]} pixels)", fontsize=8.5)
    axs[1].imshow(base)
    axs[1].set_title("Same image, retina centred; structures marked", fontsize=8.5)
    a = axs[1]
    a.add_patch(Circle(od, dd * 0.6, fill=False, ec="#ffd34d", lw=1.6))
    a.add_patch(Circle(fo, dd * 1.5, fill=False, ec="#ffffff", lw=1.2, ls="--"))
    a.plot(*fo, marker="+", color="#7dff7d", ms=10, mew=1.8)
    a.annotate("optic disc", xy=od + [-dd * 0.45, dd * 0.4], xytext=od + [-dd * 0.6, dd * 3.0], color="white", fontsize=8, ha="center",
               arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
    a.annotate("fovea (centre of\nthe macula)", xy=fo, xytext=fo + [dd * 0.2, dd * 2.9], color="white", fontsize=8, ha="center",
               arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
    a.text(*(fo + [0, -dd * 1.75]), "macula (approx.)", color="white", fontsize=7.5, ha="center")
    a.annotate("major vessels leave\nthe eye at the disc", xy=od + [0, -dd * 0.6], xytext=od + [-dd * 0.3, -dd * 2.6], color="white", fontsize=7.5, ha="center",
               arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
    for x in axs:
        x.axis("off")
    save(fig, "fundus_annotated.png")
    return {"od": od.tolist(), "fovea": fo.tolist(), "location": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in out["location"].items()}}


def fig_cover():
    from drscreen.preprocess import preprocess_plain

    im = preprocess_plain(read_rgb(EX / f"{GRADE_EXAMPLES[0]}.png"), 600)
    cv2.imwrite(str(OUT / "cover_fundus.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


def fig_grades():
    from drscreen.preprocess import preprocess_plain

    fig, axs = plt.subplots(1, 5, figsize=(7.4, 1.95))
    for g, ax in enumerate(axs):
        ax.imshow(preprocess_plain(read_rgb(EX / f"{GRADE_EXAMPLES[g]}.png"), 384))
        ax.set_title(GRADE_NAMES[g], fontsize=7.6, color=[ACC, ACC, REF, REF, REF][g])
        ax.axis("off")
    fig.text(0.5, -0.02, "Referable (grade 2 or higher) in red. Images: APTOS 2019 test split; grades are the dataset's labels.", ha="center", fontsize=7, color=INK2)
    save(fig, "grades_gallery.png")


def fig_lesions_eophtha():
    def crop_with_mask(sub, kind, pick=0, half=90):
        d = EOPHTHA / f"e_ophtha_{kind}" / f"e_optha_{kind}"
        imgs = sorted((d / kind).rglob("*.jp*g")) + sorted((d / kind).rglob("*.JPG"))
        masks = {p.stem.replace("_" + kind, "").replace(kind, ""): p for p in (d / f"Annotation_{kind}").rglob("*.png")}
        for ip in imgs:
            key = ip.stem
            mp = next((m for k, m in masks.items() if k.startswith(key) or key.startswith(k)), None)
            if mp is None:
                continue
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None or m.max() == 0:
                continue
            n, lab, stats, cent = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
            if n < 2:
                continue
            j = 1 + int(np.argsort(-stats[1:, cv2.CC_STAT_AREA])[min(pick, n - 2)]) if kind == "EX" else 1 + min(pick, n - 2)
            cx, cy = cent[j]
            im = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
            y0, x0 = int(max(0, cy - half)), int(max(0, cx - half))
            c_im, c_m = im[y0:y0 + 2 * half, x0:x0 + 2 * half], m[y0:y0 + 2 * half, x0:x0 + 2 * half]
            return c_im, c_m, ip.parent.name + "/" + ip.name
        return None, None, None

    fig, axs = plt.subplots(2, 2, figsize=(5.6, 5.4))
    for row, (kind, title) in enumerate((("MA", "Microaneurysms"), ("EX", "Hard exudates"))):
        c_im, c_m, src = crop_with_mask(None, kind, half=36 if kind == "MA" else 110)
        if c_im is None:
            for a in axs[row]:
                a.text(0.5, 0.5, "image not available", ha="center")
                a.axis("off")
            continue
        axs[row, 0].imshow(c_im)
        axs[row, 0].set_title(f"{title}: photograph (enlarged crop)", fontsize=8)
        ov = c_im.copy()
        cnts, _ = cv2.findContours((c_m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 255), 1)
        axs[row, 1].imshow(ov)
        axs[row, 1].set_title("expert annotation outlined", fontsize=8)
        for a in axs[row]:
            a.axis("off")
    fig.text(0.5, 0.0, "Source: e-ophtha database (OPHDIAT screening network, distributed for research on DR); crops enlarged.", ha="center", fontsize=6.8, color=INK2)
    save(fig, "lesions_eophtha.png")


def fig_problem_to_computation():
    fig, ax = canvas_ax(size=(7.2, 2.7))
    steps = [("Diabetes\n(long-term high\nblood glucose)", REF_S, REF), ("Damage to small\nretinal vessels", REF_S, REF), ("Visible signs:\nmicroaneurysms,\nhaemorrhages,\nexudates", REF_S, REF),
             ("Fundus\nphotograph", AMB_S, AMB), ("Array of numbers\nH × W × 3", IND_S, IND), ("Model: grade\nprobabilities\n0 – 4", ACC_S, ACC),
             ("Decision: refer,\nno referral,\nreview or retake", ACC_S, ACC)]
    w, gap = 12.0, 2.3
    for i, (t, fc, ec) in enumerate(steps):
        x = 1 + i * (w + gap)
        box(ax, x, 12, w, 16, t, fc, ec, fs=7.2)
        if i < len(steps) - 1:
            arrow(ax, (x + w + 0.3, 20), (x + w + gap - 0.3, 20))
    for x0, x1, t, c in ((1, 1 + 3 * (w + gap) - gap, "biology", REF), (1 + 3 * (w + gap), 1 + 5 * (w + gap) - gap, "imaging and computation", IND),
                         (1 + 5 * (w + gap), 1 + 7 * (w + gap) - gap, "screening output", ACC)):
        ax.plot([x0, x1], [33, 33], color=c, lw=2)
        ax.text((x0 + x1) / 2, 34.5, t, ha="center", fontsize=8, color=c, fontweight="bold")
    ax.text(1, 4, "The project answers one computational question: from a single colour fundus photograph, is referable diabetic retinopathy "
                  "(ICDR grade 2 or higher) likely?", fontsize=7.6, color=INK2)
    save(fig, "problem_to_computation.png")


# =============================================================================== Part II: images
def fig_pixels():
    from drscreen.preprocess import preprocess_plain

    plain = preprocess_plain(read_rgb(EX / f"{GRADE_EXAMPLES[0]}.png"), 384)
    gy, gx = np.unravel_index(np.argmax(cv2.GaussianBlur(plain[..., 1].astype(np.float32), (0, 0), 6)), plain.shape[:2])
    cy, cx = 192, 150  # an area with vessels, left of centre
    patch = plain[cy - 6:cy + 6, cx - 6:cx + 6]
    fig = plt.figure(figsize=(7.2, 2.9))
    a1 = fig.add_axes([0.0, 0.05, 0.30, 0.9])
    a1.imshow(plain)
    a1.add_patch(Rectangle((cx - 6, cy - 6), 12, 12, fill=False, ec="#ffd34d", lw=1.4))
    a1.set_title("384 × 384 × 3 array", fontsize=8.5)
    a1.axis("off")
    a2 = fig.add_axes([0.33, 0.05, 0.30, 0.9])
    a2.imshow(patch, interpolation="nearest")
    for k in range(13):
        a2.axhline(k - 0.5, color="white", lw=0.4)
        a2.axvline(k - 0.5, color="white", lw=0.4)
    a2.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, ec="#ffd34d", lw=1.6))
    a2.set_title("12 × 12 crop: each square is one pixel", fontsize=8.5)
    a2.axis("off")
    a3 = fig.add_axes([0.66, 0.05, 0.34, 0.9])
    a3.axis("off")
    a3.set_title("top-left 3 × 3 pixels as numbers (R, G, B)", fontsize=8.5)
    for i in range(3):
        for j in range(3):
            r, g, b = (int(v) for v in patch[i, j])
            a3.add_patch(Rectangle((j, 2 - i), 1, 1, fc=np.array([r, g, b]) / 255, ec="white", lw=2))
            a3.text(j + 0.5, 2 - i + 0.5, f"{r}\n{g}\n{b}", ha="center", va="center", fontsize=7.4, color="white" if (r + g + b) < 360 else INK)
    a3.set_xlim(0, 3)
    a3.set_ylim(0, 3)
    a3.set_aspect("equal")
    save(fig, "pixels.png")
    return patch[:3, :3].tolist()


def fig_preprocess_steps():
    from drscreen.preprocess import IMAGENET_MEAN, IMAGENET_STD, ben_graham, circle_mask, crop_black_borders, pad_to_square

    img = read_rgb(EX / f"{GRADE_EXAMPLES[2]}.png")
    c = crop_black_borders(img)
    s = pad_to_square(c)
    r = cv2.resize(s, (384, 384), interpolation=cv2.INTER_AREA)
    m = r.copy()
    m[~circle_mask(384, 0.98)] = 0
    bg = ben_graham(m)
    x = (bg.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    panels = [(img, f"1 · original\n{img.shape[1]} × {img.shape[0]} × 3"), (c, f"2 · black border cropped\n{c.shape[1]} × {c.shape[0]} × 3"),
              (s, f"3 · padded to a square\n{s.shape[1]} × {s.shape[0]} × 3"), (m, "4 · resized + circular mask\n384 × 384 × 3 (stored)"),
              (bg, "5 · Ben Graham normalisation\n384 × 384 × 3"), (None, "6 · scaled to ImageNet statistics\ntensor 3 × 384 × 384 (float)")]
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.6))
    for ax, (im, t) in zip(axs.ravel(), panels):
        if im is None:
            ax.imshow(x[..., 1], cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        else:
            ax.imshow(im)
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    save(fig, "preprocess_steps.png")
    return {"original": list(img.shape), "cropped": list(c.shape), "square": list(s.shape), "green_channel_range": [float(x[..., 1].min()), float(x[..., 1].max())]}


def fig_augment():
    from drscreen.datasets import augment_plain
    from drscreen.preprocess import preprocess_plain

    plain = preprocess_plain(read_rgb(EX / f"{GRADE_EXAMPLES[2]}.png"), 384)
    fig, axs = plt.subplots(2, 5, figsize=(7.4, 3.3))
    axs[0, 0].imshow(plain)
    axs[0, 0].set_title("stored image", fontsize=7.6)
    for k in range(4):
        axs[0, k + 1].imshow(augment_plain(plain, np.random.default_rng(10 + k), robust=False))
        axs[0, k + 1].set_title(f"random augmentation {k + 1}", fontsize=7.6)
    blur = cv2.GaussianBlur(plain, (0, 0), 3.0)
    ok, enc = cv2.imencode(".jpg", plain[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 10])
    jpg = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    f = ((np.clip(plain.astype(np.float32) / 255.0, 0, 1) ** 1.7) - 0.5) * 0.6 + 0.5
    dark = np.clip(f * 255, 0, 255).astype(np.uint8)
    over = np.clip(plain.astype(np.float32) * 1.8 + 30, 0, 255).astype(np.uint8)
    for a, (im, t) in zip(axs[1], [(plain, "(no defect)"), (blur, "blur (σ = 3)"), (jpg, "JPEG quality 10"), (dark, "dark / low contrast"), (over, "over-exposure")]):
        a.imshow(im)
        a.set_title(t, fontsize=7.6)
    for a in axs.ravel():
        a.axis("off")
    fig.text(0.01, 0.50, "standard training augmentation (top) · extra defects used only for the robust model (bottom)", fontsize=7, color=INK2)
    save(fig, "augmentation.png")


def fig_split_flow(res):
    d, sp = res["dataset"], res["dataset"]["splits"]
    fig, ax = canvas_ax(size=(7.2, 2.6))
    box(ax, 1, 13, 15, 12, f"APTOS 2019\nlabelled images\n{d['n_labelled_found']:,}", AMB_S, AMB, fs=7.6)
    box(ax, 20, 13, 15, 12, f"read, validate,\npreprocess to 384 px\n({d['n_unreadable']} unreadable)", IND_S, IND, fs=7.4)
    box(ax, 39, 13, 17, 12, f"group duplicates\n(MD5 + perceptual hash)\n{d['n_duplicate_groups']:,} groups", IND_S, IND, fs=7.4)
    box(ax, 60, 13, 15, 12, "stratified, group-aware\nsplit (seed 42)", IND_S, IND, fs=7.4)
    for i, (k, c) in enumerate((("train", ACC), ("val", AMB), ("test", REF))):
        box(ax, 80, 26 - i * 11, 18, 8.4, f"{ {'train': 'training', 'val': 'validation', 'test': 'test'}[k]}: {sp[k]['n']:,}", "white", c, fs=7.6)
        arrow(ax, (75.4, 19), (79.6, 30 - i * 11))
    for x0, x1 in ((16.4, 19.6), (35.4, 38.6), (56.4, 59.6)):
        arrow(ax, (x0, 19), (x1, 19))
    ax.text(80, 1.5, "learn weights · pick checkpoint, temperature,\nthreshold · final unbiased measurement", fontsize=6.4, color=INK2)
    ax.text(1, 5, f"{d['n_images_in_merged_groups']} images belonged to multi-image duplicate groups;\neach group was kept inside one split.", fontsize=6.8, color=INK2)
    save(fig, "split_flow.png")


# =============================================================================== Part III: machine learning
def fig_supervised():
    fig, ax = canvas_ax(size=(7.2, 2.2))
    box(ax, 1, 10, 20, 12, "examples with answers\n(photograph, grade)", AMB_S, AMB, fs=7.6)
    box(ax, 27, 10, 20, 12, "learning algorithm\nadjusts the model's\nnumbers (weights)", IND_S, IND, fs=7.6)
    box(ax, 53, 10, 18, 12, "trained model\n(fixed weights)", ACC_S, ACC, fs=7.6)
    box(ax, 77, 10, 21, 12, "prediction for a\nnew photograph", "white", ACC, fs=7.6)
    arrow(ax, (21.4, 16), (26.6, 16))
    arrow(ax, (47.4, 16), (52.6, 16))
    arrow(ax, (71.4, 16), (76.6, 16))
    ax.text(1, 26.5, "training (once, offline)", fontsize=8, color=IND, fontweight="bold")
    ax.text(53, 26.5, "inference (every new image)", fontsize=8, color=ACC, fontweight="bold")
    save(fig, "supervised.png")


def fig_fitting():
    ep = np.linspace(1, 30, 200)
    tr = 0.15 + 0.85 * np.exp(-ep / 5)
    va = 0.32 + 0.66 * np.exp(-ep / 5) + 0.0045 * np.maximum(0, ep - 12) ** 1.5
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ax.plot(ep, tr, color=ACC, lw=1.8, label="training loss")
    ax.plot(ep, va, color=REF, lw=1.8, label="validation loss")
    best = ep[np.argmin(va)]
    ax.axvline(best, color=INK2, ls="--", lw=0.9)
    ax.text(best + 0.4, 0.80, "best checkpoint\n(lowest validation loss)", fontsize=7, color=INK2, va="top")
    ax.text(3, 0.22, "under-fitting", fontsize=7.4, color=INK2)
    ax.text(21, 0.22, "over-fitting", fontsize=7.4, color=INK2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Illustrative curves (not project data)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.4)
    save(fig, "fitting_illustration.png")


def fig_convolution():
    inp = np.array([[10, 10, 10, 200, 200, 200]] * 6, float)
    inp[0:2, 3:] = 190
    k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    out = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            out[i, j] = (inp[i:i + 3, j:j + 3] * k).sum()
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [6, 3, 4]})
    for ax, arr, title, cmap in ((axs[0], inp, "input (6 × 6 grey levels)", "Greys_r"), (axs[1], k, "3 × 3 filter\n(vertical-edge detector)", "RdBu_r"),
                                 (axs[2], out, "output feature map (4 × 4)", "RdBu_r")):
        ax.imshow(arr, cmap=cmap, vmin=-arr.__abs__().max() if cmap == "RdBu_r" else 0, vmax=arr.__abs__().max() if cmap == "RdBu_r" else 255)
        for (i, j), v in np.ndenumerate(arr):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7 if arr.shape[0] > 3 else 8.5,
                    color="white" if (cmap == "Greys_r" and v < 120) or (cmap == "RdBu_r" and abs(v) > 0.6 * abs(arr).max()) else INK)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].add_patch(Rectangle((1.5, 0.5 - 1), 3, 3, fill=False, ec="#ffb000", lw=2))
    axs[2].add_patch(Rectangle((1.5, -0.5), 1, 1, fill=False, ec="#ffb000", lw=2))
    fig.text(0.5, -0.03, f"The highlighted output value is the sum of the 9 products of the highlighted input window and the filter: "
                         f"{out[0, 2]:.0f}. Large values mark the dark-to-bright edge.", ha="center", fontsize=7.2, color=INK2)
    save(fig, "convolution.png")
    return {"highlight_value": float(out[0, 2])}


def model_and_input():
    import torch

    from drscreen.models import load_checkpoint
    from drscreen.preprocess import preprocess_for_model

    model, ck = load_checkpoint(ROOT / "artifacts" / "efficientnet_b0" / "best.pt", "cpu")
    plain, inp, x = preprocess_for_model(read_rgb(EX / f"{GRADE_EXAMPLES[2]}.png"), ck["cfg"]["img_size"], True)
    return model, ck, plain, inp, torch.from_numpy(x)[None]


def fig_feature_maps():
    import torch

    model, ck, plain, inp, x = model_and_input()
    acts = {}
    h1 = model.bn1.register_forward_hook(lambda m, i, o: acts.__setitem__("stem", o.detach()))
    h2 = model.blocks[2].register_forward_hook(lambda m, i, o: acts.__setitem__("b2", o.detach()))
    with torch.no_grad():
        logits = model(x)[0].numpy()
    h1.remove()
    h2.remove()
    stem, b2 = acts["stem"][0].numpy(), acts["b2"][0].numpy()
    order = np.argsort(-stem.reshape(stem.shape[0], -1).std(1))[:6]
    order2 = np.argsort(-b2.reshape(b2.shape[0], -1).std(1))[:4]
    fig, axs = plt.subplots(2, 6, figsize=(6.8, 2.7))
    axs = axs.ravel()
    axs[11].axis("off")
    axs[0].imshow(inp)
    axs[0].set_title("model input", fontsize=7.6)
    for a, c in zip(axs[1:7], order):
        a.imshow(stem[c], cmap="magma")
        a.set_title(f"first layer, ch {c}", fontsize=7.2)
    for a, c in zip(axs[7:11], order2):
        a.imshow(b2[c], cmap="magma")
        a.set_title(f"stage 3, ch {c}", fontsize=7.2)
    for a in axs:
        a.axis("off")
    save(fig, "feature_maps.png")
    return {"stem_shape": list(stem.shape), "stage3_shape": list(b2.shape), "logits": logits.tolist()}


def fig_effnet_arch():
    import torch

    model, ck, plain, inp, x = model_and_input()
    shapes = []
    hooks = [model.bn1.register_forward_hook(lambda m, i, o: shapes.append(("stem", tuple(o.shape))))]
    for k, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(lambda m, i, o, k=k: shapes.append((f"stage{k + 1}", tuple(o.shape)))))
    hooks.append(model.bn2.register_forward_hook(lambda m, i, o: shapes.append(("head", tuple(o.shape)))))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    n_params = sum(p.numel() for p in model.parameters())
    depth = [len(b) for b in model.blocks]
    ksz = []
    for b in model.blocks:
        conv = getattr(b[0], "conv_dw", None)
        ksz.append(conv.kernel_size[0] if conv is not None else 3)
    fig, ax = canvas_ax(size=(6.8, 2.9))
    y = 14
    items = [("input", f"3 × {x.shape[2]} × {x.shape[3]}", AMB_S, AMB)]
    for name, shp in shapes:
        items.append((name, f"{shp[1]} × {shp[2]} × {shp[3]}", ACC_S if name.startswith("stage") else IND_S, ACC if name.startswith("stage") else IND))
    items.append(("pool,\ndropout,\nlinear", "5 logits", REF_S, REF))
    w, gap = 7.3, 0.95
    for i, (name, shp, fc, ec) in enumerate(items):
        x0 = 0.6 + i * (w + gap)
        hgt = 18 if name not in ("input",) else 18
        label = name
        if name.startswith("stage"):
            k = int(name[5:]) - 1
            label = f"stage {k + 1}\n{depth[k]} block{'s' if depth[k] > 1 else ''}\n{ksz[k]}×{ksz[k]} dw"
        elif name == "stem":
            label = "stem\n3×3 conv\nstride 2"
        elif name == "head":
            label = "head\n1×1 conv"
        box(ax, x0, y, w, hgt, label, fc, ec, fs=6.7)
        ax.text(x0 + w / 2, y - 3.2, shp.replace(" × ", "×"), ha="center", fontsize=6.3, color=INK2)
        if i < len(items) - 1:
            arrow(ax, (x0 + w + 0.1, y + 9), (x0 + w + gap - 0.1, y + 9), lw=0.9)
    ax.text(0.6, 3.5, f"Shapes measured by passing a real preprocessed image through the trained checkpoint (channels × height × width). "
                      f"Total parameters: {n_params / 1e6:.2f} million.", fontsize=7, color=INK2)
    ax.text(0.6, 36, "EfficientNet-B0 as used in this project (5 output classes)", fontsize=8.6, color=INK, fontweight="bold")
    save(fig, "efficientnet_b0.png")
    return {"shapes": shapes, "params": n_params, "depth": depth, "kernels": ksz}


def fig_mbconv():
    fig, ax = canvas_ax(size=(7.2, 1.9))
    steps = [("input\nC channels", "white", INK2), ("1×1 conv\nexpand ×6", IND_S, IND), ("depthwise\n3×3 or 5×5", IND_S, IND), ("squeeze-and-\nexcitation", AMB_S, AMB),
             ("1×1 conv\nproject", IND_S, IND), ("output", "white", INK2)]
    w, gap = 13.5, 3.3
    for i, (t, fc, ec) in enumerate(steps):
        x0 = 1 + i * (w + gap)
        box(ax, x0, 7, w, 11, t, fc, ec, fs=7.2)
        if i < len(steps) - 1:
            arrow(ax, (x0 + w + 0.2, 12.5), (x0 + w + gap - 0.2, 12.5))
    ax.add_patch(FancyArrowPatch((8, 18.6), (92, 18.6), connectionstyle="arc3,rad=-0.18", arrowstyle="-|>", mutation_scale=9, color=ACC, lw=1.1))
    ax.text(50, 24.2, "skip connection (added when input and output shapes match)", ha="center", fontsize=7, color=ACC)
    save(fig, "mbconv.png")


def fig_softmax():
    z = np.array([2.1, 0.3, 3.0, 0.8, -0.5])
    p1 = np.exp(z) / np.exp(z).sum()
    p2 = np.exp(z / 2) / np.exp(z / 2).sum()
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2))
    labs = ["0", "1", "2", "3", "4"]
    axs[0].bar(labs, z, color=SLA)
    axs[0].set_title("logits (raw scores)", fontsize=8.2)
    axs[1].bar(labs, p1, color=[ACC if i == 2 else "#9fb7b1" for i in range(5)])
    axs[1].set_title("softmax probabilities (T = 1)", fontsize=8.2)
    axs[2].bar(labs, p2, color=[ACC if i == 2 else "#9fb7b1" for i in range(5)])
    axs[2].set_title("after temperature scaling (T = 2)", fontsize=8.2)
    for a, v in ((axs[1], p1), (axs[2], p2)):
        a.set_ylim(0, 0.75)
        for i, pv in enumerate(v):
            a.text(i, pv + 0.02, f"{pv:.2f}", ha="center", fontsize=6.8)
    for a in axs:
        a.set_xlabel("grade", fontsize=7.5)
    fig.text(0.5, -0.14, f"Illustrative numbers. P(referable) = P(2)+P(3)+P(4) = {p1[2:].sum():.2f} at T = 1 and {p2[2:].sum():.2f} at T = 2; "
                         "the most likely grade (2) does not change.", ha="center", fontsize=7.2, color=INK2)
    save(fig, "softmax.png")
    return {"p1": p1.round(3).tolist(), "p2": p2.round(3).tolist()}


def fig_cross_entropy():
    p = np.linspace(0.01, 1, 300)
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    ax.plot(p, -np.log(p), color=IND, lw=1.8)
    for pv in (0.9, 0.5, 0.1):
        ax.plot(pv, -np.log(pv), "o", color=REF, ms=4)
        ax.annotate(f"p = {pv}: loss {-np.log(pv):.2f}", (pv, -np.log(pv)), xytext=(pv + 0.06 if pv < 0.8 else pv - 0.42, -np.log(pv) + 0.5), fontsize=7.2)
    ax.set_xlabel("probability the model gave to the correct grade")
    ax.set_ylabel("cross-entropy loss  −ln p")
    ax.set_ylim(0, 4.8)
    save(fig, "cross_entropy.png")


def fig_gradient_descent():
    w = np.linspace(-3, 3, 300)
    L = 0.35 * w ** 2 + 0.25 * np.sin(2.4 * w) + 0.4
    dL = lambda v: 0.7 * v + 0.6 * np.cos(2.4 * v)
    f = lambda v: 0.35 * v ** 2 + 0.25 * np.sin(2.4 * v) + 0.4
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)
    for ax, lr, title in ((axs[0], 0.25, "suitable learning rate"), (axs[1], 2.3, "learning rate too large")):
        ax.plot(w, L, color=INK2, lw=1.4)
        v = 2.6
        pts = [v]
        for _ in range(7):
            v = v - lr * dL(v)
            v = float(np.clip(v, -3, 3))
            pts.append(v)
        ax.plot(pts, [f(q) for q in pts], "o-", color=REF if lr > 1 else ACC, ms=3.5, lw=1)
        ax.set_title(title, fontsize=8.2)
        ax.set_xlabel("a single weight w (illustrative)")
    axs[0].set_ylabel("loss L(w)")
    save(fig, "gradient_descent.png")


def fig_training_loop():
    fig, ax = canvas_ax(size=(7.4, 3.6))
    nodes = [(2, 36, "1  next batch of 24\ntraining images"), (26, 36, "2  augment, Ben Graham,\nnormalise"), (50, 36, "3  forward pass\n(mixed precision)"),
             (74, 36, "4  class-weighted\ncross-entropy loss"), (74, 20, "5  backpropagation:\ngradients of the loss"), (50, 20, "6  clip gradients (norm 5),\nAdamW step"),
             (26, 20, "7  update learning rate\n(warm-up, then cosine)"), (2, 20, "8  more batches?\n→ back to 1")]
    for x0, y0, t in nodes:
        box(ax, x0, y0, 21, 9.5, t, IND_S, IND, fs=6.9)
    for (a, b) in ((0, 1), (1, 2), (2, 3)):
        arrow(ax, (nodes[a][0] + 21.3, 40.7), (nodes[b][0] - 0.3, 40.7))
    arrow(ax, (84.5, 35.6), (84.5, 29.9))
    for (a, b) in ((4, 5), (5, 6), (6, 7)):
        arrow(ax, (nodes[a][0] - 0.3, 24.7), (nodes[b][0] + 21.3, 24.7))
    arrow(ax, (12.5, 29.9), (12.5, 35.6))
    box(ax, 2, 3, 38, 10, "end of epoch: predict the validation images,\ncompute QWK, save last.pt (resume point)", ACC_S, ACC, fs=6.9)
    box(ax, 46, 3, 49, 10, "QWK improved → save best.pt\nno improvement for 8 epochs, or time budget reached → stop", ACC_S, ACC, fs=6.9)
    arrow(ax, (12.5, 19.6), (12.5, 13.4))
    arrow(ax, (40.4, 8), (45.6, 8))
    save(fig, "training_loop.png")


def fig_lr_schedule(res):
    cfg = res["run"]["config"]
    n_train = res["dataset"]["splits"]["train"]["n"]
    spe = n_train // cfg["batch_size"]
    total, warm = cfg["epochs"] * spe, int(cfg["warmup_epochs"] * spe)
    s = np.arange(total)
    lam = np.where(s < warm, (s + 1) / max(1, warm), 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))) * 0.99 + 0.01)
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.plot(s / spe, cfg["lr"] * lam, color=IND, lw=1.6)
    ax.axvline(res["run"]["epochs_completed"], color=REF, ls="--", lw=0.9)
    ax.text(res["run"]["epochs_completed"] - 0.4, cfg["lr"] * 0.9, f"stopped after\nepoch {res['run']['epochs_completed']}", fontsize=7, color=REF, ha="right")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title(f"Learning-rate schedule of the training run ({spe} steps per epoch)", fontsize=8.2)
    save(fig, "lr_schedule.png")
    return {"steps_per_epoch": int(spe)}


def fig_gradcam_how():
    fig, ax = canvas_ax(size=(7.4, 2.3))
    steps = [("photograph", AMB_S, AMB), ("CNN layers", IND_S, IND), ("last feature maps\nA¹ … Aᴷ (12 × 12)", IND_S, IND), ("weights αₖ = average\ngradient of the\nclass score over Aₖ", ACC_S, ACC),
             ("ReLU(Σ αₖ Aₖ)\ncoarse heat map", ACC_S, ACC), ("enlarge and overlay\non the image", REF_S, REF)]
    w, gap = 13.6, 2.6
    for i, (t, fc, ec) in enumerate(steps):
        x0 = 1 + i * (w + gap)
        box(ax, x0, 9, w, 14, t, fc, ec, fs=6.8)
        if i < len(steps) - 1:
            arrow(ax, (x0 + w + 0.2, 16), (x0 + w + gap - 0.2, 16))
    ax.text(1, 2.5, "Grad-CAM uses one forward pass and one backward pass of the trained classifier; it does not need lesion labels.", fontsize=7.2, color=INK2)
    save(fig, "gradcam_how.png")


def fig_lesion_pipeline():
    fig, ax = canvas_ax(size=(7.4, 3.0))
    top = [("fundus photo", AMB_S, AMB), ("1280-px canvas\n(retina centred)", IND_S, IND), ("U-Net segmenter\n(EfficientNet-B0\nencoder)", IND_S, IND),
           ("5 probability maps\nMA · HE · EX · SE · disc", IND_S, IND), ("threshold +\nconnected regions", IND_S, IND)]
    bot = [("disc + fovea\nlocation", IND_S, IND), ("counts per\nquadrant, near\nthe macula", ACC_S, ACC), ("ICDR rule engine\n(met / not met /\nnot assessed)", ACC_S, ACC),
           ("rule grade +\nreasons", ACC_S, ACC), ("cross-check with\nthe classifier", REF_S, REF)]
    w, gap = 16.2, 3.3
    for i, (t, fc, ec) in enumerate(top):
        x0 = 1 + i * (w + gap)
        box(ax, x0, 27, w, 11.5, t, fc, ec, fs=6.8)
        if i < len(top) - 1:
            arrow(ax, (x0 + w + 0.2, 32.7), (x0 + w + gap - 0.2, 32.7))
    for i, (t, fc, ec) in enumerate(bot):
        x0 = 1 + (len(bot) - 1 - i) * (w + gap)
        box(ax, x0, 8, w, 11.5, t, fc, ec, fs=6.8)
        if i < len(bot) - 1:
            arrow(ax, (x0 - 0.2, 13.7), (x0 - gap + 0.2, 13.7))
    arrow(ax, (1 + 4 * (w + gap) + w / 2, 26.6), (1 + 4 * (w + gap) + w / 2, 19.9))
    ax.text(1, 2.5, "The lesion path runs beside the classifier. It explains; on this project's evidence it does not make screening more accurate.", fontsize=7.2, color=INK2)
    save(fig, "lesion_pipeline.png")


# =============================================================================== Part IV: system
def fig_architecture():
    fig, ax = canvas_ax(size=(7.6, 4.6))
    H = 100 * 4.6 / 7.6
    ax.add_patch(FancyBboxPatch((1, 33), 98, H - 35, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#f4f6fa", ec=IND, lw=0.8, ls="--"))
    ax.text(2.5, H - 4.2, "OFFLINE · Kaggle notebooks on a Tesla T4 GPU (training and evaluation)", fontsize=7.8, color=IND, fontweight="bold")
    box(ax, 3, 41, 17, 10, "datasets\nAPTOS · IDRiD ·\ne-ophtha · EyePACS", AMB_S, AMB, fs=6.6)
    box(ax, 24, 41, 18, 10, "preprocess, split,\ncache (384 px)", IND_S, IND, fs=6.6)
    box(ax, 46, 41, 18, 10, "train classifiers and\nlesion models;\nfit T and threshold", IND_S, IND, fs=6.6)
    box(ax, 68, 41, 29, 10, "outputs: best.pt, calibration.json,\nmetrics.json, predictions.csv, figures", IND_S, IND, fs=6.6)
    for a, b in ((20.4, 23.6), (42.4, 45.6), (64.4, 67.6)):
        arrow(ax, (a, 46), (b, 46))
    arrow(ax, (82.5, 40.6), (82.5, 31.5))
    ax.text(84, 35.5, "download\n(Kaggle CLI)", fontsize=6.4, color=INK2)
    ax.add_patch(FancyBboxPatch((1, 1), 98, 30, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#f3f7f5", ec=ACC, lw=0.8, ls="--"))
    ax.text(2.5, 27.6, "LOCAL · the user's computer (CPU is enough)", fontsize=7.8, color=ACC, fontweight="bold")
    box(ax, 66, 15, 31, 8.5, "artifacts/ (checkpoints) · results/ (metrics)\n→ report builder → Final_Report.pdf", ACC_S, ACC, fs=6.4)
    flow = [("upload in the\nRetinaLab app", "white"), ("quality gate", ACC_S), ("preprocess\n384 px", ACC_S), ("classifier or\nensemble + T", ACC_S),
            ("threshold →\ndecision", ACC_S), ("Grad-CAM ·\nlesions · rules", ACC_S), ("explanation +\nPDF case report", "white")]
    w, gap = 12.0, 1.9
    for i, (t, fc) in enumerate(flow):
        x0 = 2.5 + i * (w + gap)
        box(ax, x0, 3.5, w, 8.5, t, fc, ACC, fs=6.3)
        if i < len(flow) - 1:
            arrow(ax, (x0 + w + 0.1, 7.7), (x0 + w + gap - 0.1, 7.7), lw=0.9)
    arrow(ax, (75, 14.6), (50, 12.4), rad=0.1)
    ax.text(57, 15.6, "loads", fontsize=6.4, color=INK2)
    save(fig, "architecture.png")


# =============================================================================== Part V: metrics
def fig_confusion_teach():
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    cells = [[("TN = 72", "correctly passed", "#e6eef7"), ("FP = 8", "false alarm", REF_S)], [("FN = 2", "missed case", REF_S), ("TP = 18", "correctly flagged", ACC_S)]]
    for i in range(2):
        for j in range(2):
            t, s, c = cells[i][j]
            ax.add_patch(Rectangle((j, 1 - i), 1, 1, fc=c, ec="white", lw=3))
            ax.text(j + 0.5, 1.5 - i + 0.08, t, ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
            ax.text(j + 0.5, 1.5 - i - 0.18, s, ha="center", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["predicted\nnot referable", "predicted\nreferable"], fontsize=8)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["actually\nreferable", "actually\nnot referable"], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Illustrative screening of 100 eyes (20 referable)", fontsize=8.6)
    save(fig, "confusion_teaching.png")


def _dists():
    rng = np.random.default_rng(3)
    neg = np.clip(rng.beta(2, 7, 4000), 0, 1)
    pos = np.clip(rng.beta(6, 2.2, 1000), 0, 1)
    return neg, pos


def fig_threshold_tradeoff():
    neg, pos = _dists()
    t = 0.5
    bins = np.linspace(0, 1, 41)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
    a = axs[0]
    a.hist(neg, bins, color="#7f9cc2", alpha=0.75, label="not referable", density=True)
    a.hist(pos, bins, color="#c9826f", alpha=0.75, label="referable", density=True)
    a.axvline(t, color=INK, lw=1.4)
    top = a.get_ylim()[1]
    a.set_ylim(0, top * 1.45)
    a.text(t + 0.02, top * 1.36, "threshold", fontsize=7.4)
    a.text(0.02, top * 1.12, "FN: referable eyes\nbelow the threshold", fontsize=6.8, color=REF)
    a.text(0.66, top * 1.12, "FP: healthy eyes\nabove the threshold", fontsize=6.8, color=IND)
    a.set_xlabel("model score for 'referable'")
    a.set_yticks([])
    a.legend(frameon=False, fontsize=7, loc="center right")
    a.set_title("Illustrative score distributions", fontsize=8.4)
    ts = np.linspace(0, 1, 101)
    sens = [(pos >= u).mean() for u in ts]
    spec = [(neg < u).mean() for u in ts]
    b = axs[1]
    b.plot(ts, sens, color=REF, lw=1.6, label="sensitivity")
    b.plot(ts, spec, color=IND, lw=1.6, label="specificity")
    b.set_xlabel("threshold")
    b.set_ylim(0, 1.02)
    b.legend(frameon=False, fontsize=7)
    b.set_title("Moving the threshold trades one for the other", fontsize=8.4)
    save(fig, "threshold_tradeoff.png")


def fig_roc_teach():
    neg, pos = _dists()
    ts = np.linspace(1, 0, 201)
    tpr = np.array([(pos >= u).mean() for u in ts])
    fpr = np.array([(neg >= u).mean() for u in ts])
    auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.plot(fpr, tpr, color=ACC, lw=1.8)
    ax.fill_between(fpr, tpr, color=ACC, alpha=0.08)
    ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=0.8)
    for u in (0.8, 0.5, 0.2):
        i = int(np.argmin(np.abs(ts - u)))
        ax.plot(fpr[i], tpr[i], "o", color=REF, ms=4)
        ax.annotate(f"threshold {u}", (fpr[i], tpr[i]), xytext=(fpr[i] + 0.06, tpr[i] - 0.1), fontsize=6.8)
    ax.text(0.45, 0.25, f"area = {auc:.2f}", fontsize=8, color=ACC)
    ax.text(0.55, 0.47, "chance", fontsize=6.8, color=INK2, rotation=40)
    ax.set_xlabel("false-positive rate (1 − specificity)")
    ax.set_ylabel("true-positive rate (sensitivity)")
    ax.set_title("Illustrative ROC curve", fontsize=8.4)
    save(fig, "roc_teaching.png")


def fig_ppv_prevalence(res):
    s = res["internal_holdout"]["screening"]
    se, sp = s["sensitivity"], s["specificity"]
    prev = np.linspace(0.01, 0.6, 300)
    ppv = se * prev / (se * prev + (1 - sp) * (1 - prev))
    npv = sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev)
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    ax.plot(prev, ppv, color=REF, lw=1.7, label="PPV (a 'refer' result is right)")
    ax.plot(prev, npv, color=IND, lw=1.7, label="NPV (a 'no referral' result is right)")
    ax.axvline(s["prevalence"], color=INK2, ls="--", lw=0.9)
    ax.text(s["prevalence"] + 0.01, 0.06, f"APTOS test set\nprevalence {s['prevalence']:.2f}", fontsize=7, color=INK2)
    ax.axvline(0.10, color=AMB, ls=":", lw=1)
    ax.text(0.105, 0.42, "a population with\n10% referable\n(illustrative)", fontsize=6.8, color=AMB)
    ax.set_xlabel("prevalence of referable DR")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="center right")
    ax.set_title(f"Same sensitivity ({se:.3f}) and specificity ({sp:.3f}), different populations", fontsize=8)
    save(fig, "ppv_prevalence.png")
    p10 = 0.10
    return {"ppv_at_10pct": float(se * p10 / (se * p10 + (1 - sp) * (1 - p10))), "npv_at_10pct": float(sp * (1 - p10) / (sp * (1 - p10) + (1 - se) * p10))}


def fig_bootstrap(res):
    p = pd.read_csv(ROOT / "results" / "predictions.csv")
    t = p[p.split == "test"]
    y, yhat = t.true_referable.to_numpy(), t.pred_referable.to_numpy()
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        yy, pp = y[idx], yhat[idx]
        vals.append(((yy == 1) & (pp == 1)).sum() / max(1, (yy == 1).sum()))
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    point = ((y == 1) & (yhat == 1)).sum() / (y == 1).sum()
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    ax.hist(vals, bins=30, color="#9fb7b1", ec="white")
    for v, c, l in ((lo, REF, f"2.5th percentile {lo:.3f}"), (hi, REF, f"97.5th percentile {hi:.3f}"), (point, INK, f"measured {point:.3f}")):
        ax.axvline(v, color=c, lw=1.3, ls="--" if c == REF else "-")
    ax.text(lo, ax.get_ylim()[1] * 0.92, f"{lo:.3f}", color=REF, fontsize=7, ha="right")
    ax.text(hi, ax.get_ylim()[1] * 0.92, f" {hi:.3f}", color=REF, fontsize=7)
    ax.set_xlabel("sensitivity in a bootstrap resample of the 519 test images")
    ax.set_yticks([])
    ax.set_title("2,000 bootstrap resamples of the saved test predictions", fontsize=8.2)
    save(fig, "bootstrap.png")
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def fig_dice_iou():
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((3.6, 2.6), 1.9, fc=ACC, alpha=0.35, ec=ACC, lw=1.4))
    ax.add_patch(Circle((5.2, 2.6), 1.9, fc=REF, alpha=0.35, ec=REF, lw=1.4))
    ax.text(2.4, 2.6, "expert\nmask A", ha="center", va="center", fontsize=7.6)
    ax.text(6.5, 2.6, "model\nmask B", ha="center", va="center", fontsize=7.6)
    ax.text(4.4, 2.6, "A∩B", ha="center", va="center", fontsize=7.6, fontweight="bold")
    ax.text(7.9, 3.4, "Dice = 2|A∩B| / (|A|+|B|)", fontsize=7.6)
    ax.text(7.9, 2.6, "IoU = |A∩B| / |A∪B|", fontsize=7.6)
    ax.text(7.9, 1.8, "1 = perfect overlap\n0 = no overlap", fontsize=7, color=INK2)
    save(fig, "dice_iou.png")


def fig_mental_model():
    fig, ax = canvas_ax(size=(7.6, 3.3))
    cols = [("BIOLOGY", REF, ["diabetes damages\nretinal vessels", "lesions appear;\nICDR grades 0–4", "referable =\ngrade ≥ 2"]),
            ("IMAGE", AMB, ["fundus photograph", "crop · mask · resize\nBen Graham", "tensor\n3 × 384 × 384"]),
            ("MODEL", IND, ["EfficientNet\n(transfer learning)", "softmax → grade\nprobabilities", "temperature-scaled\nreferable score"]),
            ("EVALUATION", ACC, ["threshold chosen on\nvalidation (sens ≥ 0.90)", "test: sensitivity,\nspecificity, AUC, QWK", "bootstrap CIs;\nexternal IDRiD check"]),
            ("OUTPUT", SLA, ["quality gate →\nrefer / no referral /\nreview / retake", "Grad-CAM + lesion\nevidence + rules", "RetinaLab app +\nPDF case report"])]
    w = 18.4
    for i, (name, c, items) in enumerate(cols):
        x0 = 1 + i * (w + 1.4)
        ax.text(x0 + w / 2, 41.5, name, ha="center", fontsize=8.4, color=c, fontweight="bold")
        for j, t in enumerate(items):
            box(ax, x0, 29 - j * 12.6, w, 9.6, t, "white", c, fs=6.5)
            if j < 2:
                arrow(ax, (x0 + w / 2, 28.6 - j * 12.6), (x0 + w / 2, 26.1 - j * 12.6), color=c, lw=0.9)
        if i < len(cols) - 1:
            arrow(ax, (x0 + w + 0.1, 20), (x0 + w + 1.3, 20), lw=1.0)
    save(fig, "mental_model.png")


WALK_IMAGE = "0318598cfd16"  # APTOS test image (dataset grade 4); also the interface's demonstration image


def fig_walkthrough():
    """Run the application's real Predictor (default ensemble) on one stored test image; save its images and every number for Part VII."""
    import pandas as pd

    from drscreen.explain import confidence, verdict
    from drscreen.predict import Predictor

    ens = ROOT / "artifacts" / "ensemble" / "ensemble.json"
    pred = Predictor(ens, ensemble=ens, device="cpu") if ens.is_file() else Predictor(ROOT / "artifacts" / "efficientnet_b0_robust" / "best.pt", device="cpu")
    img = read_rgb(EX / f"{WALK_IMAGE}.png")
    r = pred.predict(img, with_gradcam=True)
    panels = [(r.original, "1 · uploaded photograph"), (r.processed, "2 · preprocessed (384 px)"), (r.gradcam, "3 · Grad-CAM (first member)"),
              (r.lesion_overlay, "4 · lesion detections, disc, fovea")]
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.2))
    for a, (im, t) in zip(axs, panels):
        if im is not None:
            a.imshow(im)
        a.set_title(t, fontsize=7.4)
        a.axis("off")
    save(fig, "walkthrough_images.png")
    probs = list(r.grade_probs or [])
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    ax.bar([str(k) for k in range(5)], probs, color=[ACC if k == r.grade else "#9fb7b1" for k in range(5)])
    for k, pv in enumerate(probs):
        ax.text(k, pv + 0.02, f"{pv:.2f}", ha="center", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("grade")
    ax.set_title("Grade probabilities for this image (ensemble, temperature-scaled)", fontsize=7.8)
    save(fig, "walkthrough_probs.png")
    truth = pd.read_csv(ROOT / "results" / "predictions.csv").set_index("image_id").loc[WALK_IMAGE]
    ev = r.lesion_evidence or {}
    f = ev.get("features", {})
    rule = ev.get("rule", {})
    conf, _ = confidence(r)
    facts = {"image_id": WALK_IMAGE, "dataset_grade": int(truth["true_grade"]), "width": int(img.shape[1]), "height": int(img.shape[0]),
             "quality_status": (r.quality or {}).get("status"), "quality_reason": (r.quality or {}).get("reason"), "model": r.model_name,
             "grade": int(r.grade), "grade_name": r.grade_name, "probs": [float(v) for v in probs], "p_referable": float(r.p_referable),
             "threshold": float(r.threshold), "review_margin": float(getattr(r, "review_margin", 0.0)), "decision": (r.decision or {}).get("action"),
             "verdict": verdict(r)["title"], "confidence": conf, "lesions_available": bool(ev.get("available")),
             "counts": {k: int(f.get(f"{k}_count", 0)) for k in ("MA", "HE", "EX", "SE")}, "ex_near_macula": int(f.get("EX_within_1DD", 0)),
             "rule_grade": rule.get("grade"), "cross_check": (r.cross_check or {}).get("status"), "cross_check_message": (r.cross_check or {}).get("message"),
             "latency_s": round((r.latency_ms or 0) / 1000, 1)}
    (OUT / "walkthrough.json").write_text(json.dumps(facts, indent=1))
    return facts


def main():
    res = json.loads((ROOT / "results" / "metrics.json").read_text())
    facts = {}
    fig_eye_anatomy()
    fig_retina_layers()
    facts["fundus_locator"] = fig_fundus_annotated()
    fig_cover()
    fig_grades()
    fig_lesions_eophtha()
    fig_problem_to_computation()
    facts["pixels_3x3"] = fig_pixels()
    facts["preprocess"] = fig_preprocess_steps()
    fig_augment()
    fig_split_flow(res)
    fig_supervised()
    fig_fitting()
    facts["convolution"] = fig_convolution()
    facts["feature_maps"] = fig_feature_maps()
    facts["effnet"] = fig_effnet_arch()
    fig_mbconv()
    facts["softmax"] = fig_softmax()
    fig_cross_entropy()
    fig_gradient_descent()
    fig_training_loop()
    facts["lr"] = fig_lr_schedule(res)
    fig_gradcam_how()
    fig_lesion_pipeline()
    fig_architecture()
    fig_confusion_teach()
    fig_threshold_tradeoff()
    fig_roc_teach()
    facts["ppv"] = fig_ppv_prevalence(res)
    facts["bootstrap"] = fig_bootstrap(res)
    fig_dice_iou()
    fig_mental_model()
    facts["walkthrough"] = fig_walkthrough()
    (OUT / "figure_facts.json").write_text(json.dumps(facts, indent=1, default=str))
    print(json.dumps({k: v for k, v in facts.items() if k in ("effnet", "bootstrap", "ppv", "preprocess", "lr", "convolution")}, default=str)[:2500])


if __name__ == "__main__":
    main()
