"""Generate notebooks/M00_setup.ipynb, M01_preprocessing.ipynb, M02_training_and_evaluation.ipynb (not executed).

    python scripts/make_notebooks.py

Notebooks call the drscreen package (same code as the Kaggle run). They run on Kaggle (data under /kaggle/input) or locally
(set DATA_ROOT). M02 reads the saved results of the remote run; it does not train anything itself unless RUN_TRAINING = True.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''import sys, os
from pathlib import Path
ROOT = Path.cwd()
for cand in (ROOT, ROOT.parent, ROOT.parent.parent):
    if (cand / "src" / "drscreen").is_dir():
        sys.path.insert(0, str(cand / "src")); ROOT = cand; break
# On Kaggle the package is embedded in the kernel script; for notebooks, upload src/drscreen as a Dataset or copy it next to the notebook.
DATA_ROOT = os.environ.get("DR_DATA_ROOT") or ("/kaggle/input/aptos2019-blindness-detection" if Path("/kaggle/input").exists() else str(ROOT / "data" / "aptos2019"))
print("project root:", ROOT, "| data root:", DATA_ROOT)'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return n


M00 = nb([
    ("md", "# M00: Setup, images as arrays, class distribution\n\n> Academic research prototype; outputs are not a medical diagnosis.\n\nChecks the environment, loads the APTOS 2019 labels, shows one image per grade, splits an image into R/G/B channels and plots the class distribution. **Not executed in the repository; run it on Kaggle or locally with the data present.**"),
    ("code", SETUP),
    ("code", "import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt\nprint('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"),
    ("code", "from drscreen.datasets import load_labelled_dataset\nfrom drscreen.preprocess import read_image\ndf = load_labelled_dataset(DATA_ROOT, 'APTOS 2019')\nprint(len(df), 'labelled images; dropped:', df.attrs['dropped'])\nprint(df['grade'].value_counts().sort_index())\nprint('referable prevalence (grade>=2): %.3f' % (df['grade'] >= 2).mean())"),
    ("code", "rng = np.random.default_rng(0)\nsample = df.sample(10, random_state=0)\nfor r in sample.itertuples():\n    im = read_image(r.path)\n    print(r.image_id, 'grade', r.grade, 'shape', im.shape, 'dtype', im.dtype, 'min/max', im.min(), im.max())"),
    ("code", "from drscreen.metrics import GRADE_NAMES\nfig, axs = plt.subplots(2, 5, figsize=(15, 6))\nfor g in range(5):\n    r = df[df.grade == g].iloc[0]\n    im = read_image(r.path)\n    axs[0, g].imshow(im); axs[0, g].set_title(GRADE_NAMES[g]); axs[0, g].axis('off')\n    axs[1, g].imshow(im[..., 1], cmap='gray'); axs[1, g].set_title('green channel'); axs[1, g].axis('off')\nplt.tight_layout(); plt.show()\n# Test: the green channel shows vessels and red lesions with the highest contrast (haemoglobin absorbs green light)."),
    ("code", "ax = df['grade'].value_counts().sort_index().plot.bar(figsize=(5, 3), title='APTOS 2019: images per grade'); ax.set_xlabel('ICDR grade'); plt.show()"),
])

M01 = nb([
    ("md", "# M01: Preprocessing\n\nCrop black border → pad square → resize → circular FOV mask → Ben Graham colour normalisation. Tests: every output has the target size, no crash on 200 random images, before/after grid looks consistent. **Not executed in the repository.**"),
    ("code", SETUP),
    ("code", "import numpy as np, matplotlib.pyplot as plt, time\nfrom drscreen.datasets import load_labelled_dataset\nfrom drscreen.preprocess import read_image, preprocess_plain, to_model_input, ImageError\ndf = load_labelled_dataset(DATA_ROOT)\nSIZE = 384\nsample = df.sample(min(200, len(df)), random_state=0)\nbad, t0 = [], time.time()\nfor r in sample.itertuples():\n    try:\n        out = preprocess_plain(read_image(r.path), SIZE)\n        assert out.shape == (SIZE, SIZE, 3) and out.dtype == np.uint8\n    except (ImageError, AssertionError) as e:\n        bad.append((r.image_id, str(e)))\nprint(f'{len(sample)-len(bad)}/{len(sample)} images OK, {len(bad)} failures, {(time.time()-t0)/len(sample)*1000:.0f} ms/image'); print(bad[:5])"),
    ("code", "fig, axs = plt.subplots(3, 6, figsize=(15, 8))\nfor j, r in enumerate(sample.head(6).itertuples()):\n    raw = read_image(r.path); plain = preprocess_plain(raw, SIZE)\n    for i, (im, t) in enumerate(((raw, 'original'), (plain, 'cropped/masked'), (to_model_input(plain), 'Ben Graham'))):\n        axs[i, j].imshow(im); axs[i, j].set_title(f'{t} (grade {r.grade})', fontsize=8); axs[i, j].axis('off')\nplt.tight_layout(); plt.show()"),
])

M02 = nb([
    ("md", "# M02: Training and evaluation: results of the remote Kaggle run\n\nThis notebook **reads and presents** the saved outputs of the Kaggle run (`results/metrics.json`, `results/predictions.csv`, figures). The training itself ran remotely via `notebooks/kaggle/train_and_evaluate.py`. Nothing below is recomputed or hand-edited.\n\n> Academic research prototype; outputs are not a medical diagnosis."),
    ("code", SETUP + "\nimport json\nfrom IPython.display import display, Image, Markdown\nRES = ROOT / 'results'; FIG = ROOT / 'report' / 'figures'\nm = json.loads((RES / 'metrics.json').read_text())\nprint('run:', m['environment'].get('gpu'), '| epochs completed:', m['run']['epochs_completed'], '| stop reason:', m['run']['stop_reason'])"),
    ("code", "from drscreen.metrics import metrics_table_rows\nfrom drscreen.report import md_table\nh = m['internal_holdout']; s = h['screening']\nprint(h['dataset'], '|', h['evaluation_type'], '| n =', h['n_samples'], '| positives =', s['n_positive'], '| prevalence = %.3f' % s['prevalence'], '| threshold = %.3f (chosen on validation)' % s['threshold'])\ndisplay(Markdown(md_table(['Metric','Measured result','95% CI','What it means','Why it matters'], metrics_table_rows(h))))"),
    ("code", "display(Markdown(f\"TP={s['tp']}  FP={s['fp']}  TN={s['tn']}  FN={s['fn']}  (FN = missed referable cases; FP = false alarms)\"))\nfor f in ['confusion_referable.png','confusion_5grade.png','roc_pr.png','reliability.png','learning_curves.png']:\n    if (FIG / f).exists(): display(Image(str(FIG / f)))"),
    ("code", "for f in ['gradcam_examples.png','error_examples.png','quality_degradation.png','ablation_auc.png']:\n    if (FIG / f).exists(): display(Image(str(FIG / f)))\nprint('Ablation:'); display(Markdown(md_table(['Configuration','N','Sens.','Spec.','AUC','QWK'], [[r['name'], r['n'], r.get('sensitivity'), r.get('specificity'), r.get('roc_auc'), r.get('qwk')] for r in m.get('ablation', [])])))"),
    ("code", "RUN_TRAINING = False  # training is meant to run on Kaggle's GPU; set True only on a GPU machine with the data present\nif RUN_TRAINING:\n    from drscreen.pipeline import Run, RunConfig\n    Run(RunConfig(data_root=DATA_ROOT, work_dir=str(ROOT / 'outputs'))).run_all()"),
])


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, n in (("M00_setup", M00), ("M01_preprocessing", M01), ("M02_training_and_evaluation", M02)):
        nbf.write(n, out / f"{name}.ipynb")
        print("wrote", out / f"{name}.ipynb")


if __name__ == "__main__":
    main()
