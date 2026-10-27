"""End-to-end experiment pipeline (data -> train -> evaluate -> baselines -> quality gate -> Grad-CAM -> latency -> external).

The SAME code runs on Kaggle (real data, GPU) and locally in smoke mode (synthetic data). Each stage is isolated:
an optional stage that fails is recorded in run_status.json with its traceback and the run continues, so the core
results are always saved. Nothing here fabricates numbers: every value in metrics.json comes from a computation on
the evaluated images, and metrics.json records which stages ran.
"""
from __future__ import annotations

import dataclasses
import json
import platform
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from . import datasets as D
from . import metrics as M
from . import plots as P
from .fusion import choose_review_margin, pipeline_metrics
from .gradcam import GradCAM, overlay_heatmap, randomization_sanity_check
from .models import grade_probs, load_checkpoint, referable_prob
from .preprocess import circle_mask, preprocess_for_model, read_image, to_model_input, to_tensor_array
from .quality import DEGRADATIONS, QualityThresholds, assess_quality, compute_quality_metrics, degrade, enhance_image, fit_thresholds
from .train import TrainConfig, make_loader, predict_logits, train, verify_checkpoint_roundtrip


@dataclass
class RunConfig:
    data_root: str | None = None
    external_root: str | None = None
    work_dir: str = "outputs"
    cache_dir: str = "/tmp/drcache"
    dataset_name: str = "APTOS 2019"
    smoke: bool = False
    smoke_n: int = 240
    img_size: int = 384
    epochs: int = 30
    batch_size: int = 24
    lr: float = 3e-4
    num_workers: int = 4
    arch: str = "efficientnet_b0"
    pretrained: bool = True
    pretrained_file: str | None = None
    seed: int = 42
    target_sensitivity: float = 0.90
    max_minutes: float | None = None
    n_boot: int = 1000
    run_baselines: bool = True
    baseline_epochs: int = 12
    baseline_img_size: int = 224
    run_quality: bool = True
    quality_eval_n: int = 150
    run_gradcam: bool = True
    resume_search_root: str | None = None
    robust_aug: bool = False
    run_name: str | None = None  # artifact folder name (default: the architecture name)
    expected_n_labelled: int | None = None  # integrity guard: abort if the labelled-image count differs (catches wrong label files)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Run:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.out = Path(cfg.work_dir)
        self.fig = self.out / "figures"
        self.res = self.out / "results"
        for d in (self.out, self.fig, self.res, self.out / "artifacts"):
            d.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.status: dict = {}
        self.metrics: dict = {"config": dataclasses.asdict(cfg), "stages": self.status}
        self.s: dict = {}  # shared state between stages
        self.t0 = time.time()

    # ----------------------------------------------------------------------------- infrastructure
    def stage(self, name: str, fn, critical: bool = False) -> bool:
        t = time.time()
        log(f"=== stage: {name}")
        try:
            fn()
            self.status[name] = {"status": "ok", "seconds": round(time.time() - t, 1)}
            ok = True
        except Exception as e:
            tb = traceback.format_exc()
            log(f"!!! stage {name} FAILED: {e}\n{tb}")
            self.status[name] = {"status": "failed", "seconds": round(time.time() - t, 1), "error": f"{type(e).__name__}: {e}", "traceback": tb[-3000:]}
            ok = False
            if critical:
                self.save()
                raise
        self.save()
        return ok

    def save(self):
        M.save_json(self.metrics, self.res / "metrics.json")
        M.save_json(self.status, self.res / "run_status.json")

    def budget_left_min(self) -> float | None:
        return None if not self.cfg.max_minutes else self.cfg.max_minutes - (time.time() - self.t0) / 60

    # ----------------------------------------------------------------------------- stages
    def run_all(self):
        c = self.cfg
        self.stage("environment", self.s_env)
        self.stage("data", self.s_data, critical=True)
        self.stage("train", self.s_train, critical=True)
        self.stage("evaluate", self.s_eval, critical=True)
        if c.run_baselines:
            self.stage("baselines", self.s_baselines)
        if c.run_quality:
            self.stage("quality_gate", self.s_quality)
        if c.run_gradcam:
            self.stage("gradcam", self.s_gradcam)
        self.stage("latency", self.s_latency)
        if c.external_root:
            self.stage("external_validation", self.s_external)
        self.stage("ablation", self.s_ablation)
        self.metrics["total_minutes"] = round((time.time() - self.t0) / 60, 1)
        self.save()
        log(f"pipeline finished in {self.metrics['total_minutes']} min")

    def s_env(self):
        import sklearn

        env = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
               "opencv": cv2.__version__, "sklearn": sklearn.__version__, "cuda_available": torch.cuda.is_available()}
        try:
            import timm

            env["timm"] = timm.__version__
        except Exception:
            pass
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            env["cuda_runtime"] = torch.version.cuda
        else:
            env["gpu"] = "none (CPU run)"
        self.metrics["environment"] = env
        log(env)

    def s_data(self):
        c = self.cfg
        df = D.load_labelled_dataset(c.data_root, c.dataset_name)
        dropped = df.attrs.get("dropped", {})
        label_sources = df["source_csv"].value_counts().to_dict()
        log("label tables used:", label_sources)
        if c.smoke and len(df) > c.smoke_n:
            df = df.groupby("grade", group_keys=False).apply(lambda g: g.sample(min(len(g), max(6, c.smoke_n // 5)), random_state=c.seed)).reset_index(drop=True)
        log(f"{len(df)} labelled images found; dropped: {dropped}")
        if c.expected_n_labelled and not c.smoke and len(df) != c.expected_n_labelled:
            raise RuntimeError(f"data integrity check failed: found {len(df)} labelled images but expected {c.expected_n_labelled}; label tables used: {label_sources}")
        df_ok, cache_path = D.build_cache(df, c.img_size, c.cache_dir)
        rej = len(df) - len(df_ok)
        df_ok = D.make_splits(df_ok, seed=c.seed)
        assert (df_ok["cache_idx"].to_numpy() == np.arange(len(df_ok))).all()
        D.save_manifest(df_ok, self.out / "artifacts" / "split_manifest.csv", {"seed": c.seed, "img_size": c.img_size})
        self.s.update(df=df_ok, cache=np.load(cache_path, mmap_mode="r"), labels=df_ok["grade"].to_numpy())
        groups = df_ok["group"].nunique()
        splits = D.split_summary(df_ok)
        self.metrics["dataset"] = {
            "name": c.dataset_name, "label_tables_used": label_sources, "n_labelled_found": int(len(df)), "n_unreadable": int(rej), "dropped_before_loading": dropped,
            "n_duplicate_groups": int(groups), "n_images_in_merged_groups": int(len(df_ok) - groups), "splits": splits,
            "grade_counts_total": {str(g): int((df_ok["grade"] == g).sum()) for g in range(5)}, "img_size": c.img_size,
            "split_method": "stratified on grade, group-aware (exact + near-duplicate photos kept together), seed %d" % c.seed,
            "patient_level_split": False,
            "limitation": "APTOS 2019 provides no patient/eye identifiers, so patient-level (left/right eye) separation could not be enforced; "
                          "only exact/near-duplicate image grouping was used. Test results may be optimistic if both eyes of a patient are present.",
            "smoke_subset": bool(c.smoke),
        }
        P.plot_class_distribution({k: [v["grade_counts"][str(i)] for i in range(5)] for k, v in splits.items()}, self.fig / "class_distribution.png")
        # one example per grade: original / processed / Ben Graham / green channel
        rows, titles = [[], [], [], []], []
        for g in range(5):
            r = df_ok[df_ok["grade"] == g].iloc[0]
            raw = read_image(r["path"])
            plain = np.asarray(self.s["cache"][int(r["cache_idx"])])
            rows[0].append(raw)
            rows[1].append(plain)
            rows[2].append(to_model_input(plain, True))
            rows[3].append(cv2.cvtColor(plain[..., 1], cv2.COLOR_GRAY2RGB))
            titles.append(M.GRADE_NAMES[g])
        grid_titles = titles + [""] * 15
        P.plot_image_grid([im for row in rows for im in row], grid_titles, self.fig / "preprocessing_examples.png", ncols=5, figw=2.2,
                          suptitle="Rows: original | cropped+masked+resized | Ben Graham normalised (model input) | green channel")

    def s_train(self):
        c = self.cfg
        tcfg = TrainConfig(arch=c.arch, img_size=c.img_size, epochs=c.epochs, batch_size=c.batch_size, lr=c.lr, num_workers=c.num_workers, seed=c.seed,
                           pretrained=c.pretrained, pretrained_file=c.pretrained_file, max_minutes=c.max_minutes, out_dir=str(self.out / "artifacts"),
                           run_name=c.run_name or c.arch, max_train_batches=2 if c.smoke else None, robust_aug=c.robust_aug)
        self.s["tcfg"] = tcfg
        run_dir = Path(tcfg.out_dir) / tcfg.run_name
        if c.resume_search_root and not (run_dir / "last.pt").exists():
            for p in Path(c.resume_search_root).rglob("last.pt"):
                if p.parent.name == tcfg.run_name:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "last.pt").write_bytes(p.read_bytes())
                    log(f"found previous checkpoint to resume from: {p}")
                    break
        summ = train(tcfg, self.s["cache"], self.s["df"], self.device, log)
        self.metrics["run"] = {**summ, "config": tcfg.to_dict()}
        hist = pd.read_csv(run_dir / "history.csv")
        hist.to_csv(self.res / "training_history.csv", index=False)
        P.plot_learning_curves(hist.to_dict("records"), self.fig / "learning_curves.png", f"{c.arch}: training history ({len(hist)} epochs)")

    def _logits_for(self, model, split: str):
        df, tcfg = self.s["df"], self.s["tcfg"]
        idx = df.loc[df["split"] == split, "cache_idx"].to_numpy()
        loader = make_loader(self.s["cache"], self.s["labels"], idx, tcfg, False, self.device)
        lg, y = predict_logits(model, loader, self.device, tcfg.amp)
        return lg, y, df[df["split"] == split]

    def s_eval(self):
        c, tcfg = self.cfg, self.s["tcfg"]
        best = Path(tcfg.out_dir) / tcfg.run_name / "best.pt"
        model, ck = load_checkpoint(best, self.device)
        self.s["model"] = model
        te_idx = self.s["df"].loc[self.s["df"]["split"] == "test", "cache_idx"].to_numpy()[:16]
        self.metrics["checkpoint_reload_check"] = verify_checkpoint_roundtrip(best, self.s["cache"], self.s["labels"], te_idx, tcfg, "cpu")
        lv, yv, dv = self._logits_for(model, "val")
        lt, yt, dt = self._logits_for(model, "test")
        T = M.fit_temperature(lv, yv)
        pv_u, pv_c = grade_probs(lv), grade_probs(lv, T)
        pt_u, pt_c = grade_probs(lt), grade_probs(lt, T)
        thr = M.select_threshold(M.referable_labels(yv), M.referable_prob(pv_c), c.target_sensitivity)
        margin = choose_review_margin(yv, M.referable_prob(pv_c), thr["threshold"], 0.10)
        self.metrics["validation_threshold_selection"] = {**thr, "temperature": T, "review_margin": margin,
                                                          "val_nll_before": M.nll(lv, yv), "val_nll_after": M.nll(lv, yv, T)}
        self.s.update(T=T, thr=thr["threshold"], margin=margin, yv=yv, yt=yt, pv_c=pv_c, pt_c=pt_c, pt_u=pt_u, dv=dv, dt=dt, lt=lt)
        hold = M.evaluate_predictions(yt, pt_c, thr["threshold"], f"{c.dataset_name} internal test split", "internal holdout (same source as training)",
                                      c.n_boot, c.seed, probs5_uncal=pt_u)
        self.metrics["internal_holdout"] = hold
        self.metrics["internal_holdout_default_threshold_0.5"] = M.binary_metrics(M.referable_labels(yt), M.referable_prob(pt_c), 0.5)
        self.metrics["validation_split_reference"] = M.evaluate_predictions(yv, pv_c, thr["threshold"], f"{c.dataset_name} validation split",
                                                                           "validation (used for threshold/temperature/checkpoint selection; NOT unbiased)", 200, c.seed)
        # calibration file used by the Predictor / app
        cal = {"temperature": T, "threshold": thr["threshold"], "review_margin": margin, "threshold_rule": thr["rule"], "fitted_on": "validation split"}
        (Path(tcfg.out_dir) / tcfg.run_name / "calibration.json").write_text(json.dumps(cal, indent=2))
        # predictions.csv
        rows = []
        for split, d, y, pc in (("val", dv, yv, pv_c), ("test", dt, yt, pt_c)):
            pr = M.referable_prob(pc)
            t = pd.DataFrame({"image_id": d["image_id"].to_numpy(), "split": split, "true_grade": y, "pred_grade": pc.argmax(1),
                              **{f"p_grade{k}": pc[:, k] for k in range(5)}, "p_referable": pr, "true_referable": M.referable_labels(y),
                              "pred_referable": (pr >= thr["threshold"]).astype(int)})
            t["grade_correct"] = t["true_grade"] == t["pred_grade"]
            rows.append(t)
        pd.concat(rows).to_csv(self.res / "predictions.csv", index=False)
        s = hold["screening"]
        P.plot_binary_confusion(s, self.fig / "confusion_referable.png")
        P.plot_grade_confusion(hold["grades"]["confusion_matrix"], self.fig / "confusion_5grade.png")
        P.plot_roc_pr(yt, pt_c, thr["threshold"], self.fig / "roc_pr.png", "EfficientNet-B0")
        P.plot_reliability(yt, pt_u, pt_c, self.fig / "reliability.png")
        P.plot_threshold_sweep(yv, pv_c, thr["threshold"], self.fig / "threshold_sweep_validation.png")
        log(f"TEST referable: sens {s['sensitivity']:.3f} spec {s['specificity']:.3f} auc {s['roc_auc']:.3f} | QWK {hold['grades']['qwk']:.3f} | T={T:.2f} thr={thr['threshold']:.3f}")

    # ----------------------------------------------------------------------------- baselines
    @staticmethod
    def handcrafted_features(plain: np.ndarray) -> np.ndarray:
        im = cv2.resize(plain, (128, 128), interpolation=cv2.INTER_AREA)
        m = circle_mask(128, 0.95)
        g = im[..., 1]
        feats = []
        for ch in range(3):
            v = im[..., ch][m].astype(float)
            feats += [v.mean(), v.std()]
        hist = np.histogram(g[m], bins=16, range=(0, 255))[0].astype(float)
        feats += list(hist / hist.sum())
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        feats += [(bh[m] > t).mean() for t in (6, 12, 24)] + [(th[m] > t).mean() for t in (10, 20, 40)]
        feats += [float(cv2.Laplacian(g, cv2.CV_64F).var())]
        return np.array(feats, np.float32)

    def s_baselines(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        c, df, cache = self.cfg, self.s["df"], self.s["cache"]
        tr, va, te = (df.index[df["split"] == s].to_numpy() for s in ("train", "val", "test"))
        yv_b = M.referable_labels(self.s["yv"])
        out, store = {}, {}

        def record(name, pv, pt):
            thr = M.select_threshold(yv_b, M.referable_prob(pv), c.target_sensitivity)["threshold"]
            yt = self.s["yt"]
            sb = M.binary_metrics(M.referable_labels(yt), M.referable_prob(pt), thr)
            out[name] = {"sensitivity": sb["sensitivity"], "specificity": sb["specificity"], "roc_auc": sb["roc_auc"], "pr_auc": sb["pr_auc"],
                         "qwk": M.grade_metrics(yt, pt.argmax(1))["qwk"], "threshold": thr, "n": int(len(yt)), "tp": sb["tp"], "fp": sb["fp"], "tn": sb["tn"], "fn": sb["fn"]}
            store[name] = (pt, thr)

        # (a) majority-class prior
        prior = np.bincount(self.s["labels"][tr], minlength=5) / len(tr)
        record("Majority-class prior (no image information)", np.tile(prior, (len(va), 1)), np.tile(prior, (len(te), 1)))
        # (b) hand-crafted image statistics + gradient boosting
        X = np.stack([self.handcrafted_features(np.asarray(cache[i])) for i in range(len(df))])
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, class_weight="balanced", random_state=c.seed)
        gb.fit(X[tr], self.s["labels"][tr])

        def full(pr):
            o = np.zeros((len(pr), 5))
            o[:, gb.classes_] = pr
            return o

        record("Hand-crafted features + gradient boosting", full(gb.predict_proba(X[va])), full(gb.predict_proba(X[te])))
        # (c) CNN from scratch
        scfg = TrainConfig(arch="small_cnn", img_size=c.baseline_img_size, epochs=c.baseline_epochs, batch_size=32, lr=1e-3, num_workers=c.num_workers, seed=c.seed,
                           pretrained=False, out_dir=str(self.out / "artifacts"), run_name="small_cnn", patience=6, max_train_batches=2 if c.smoke else None,
                           max_minutes=(min(25, self.budget_left_min() * 0.15) if self.budget_left_min() else 25))
        _, p224 = D.build_cache(df[["image_id", "grade", "path"]], c.baseline_img_size, Path(c.cache_dir) / "b224")
        cache224 = np.load(p224, mmap_mode="r")
        summ = train(scfg, cache224, df, self.device, log)
        m2, _ = load_checkpoint(Path(scfg.out_dir) / "small_cnn" / "best.pt", self.device)
        outs = []
        for split in ("val", "test"):
            idx = df.loc[df["split"] == split, "cache_idx"].to_numpy()
            ld = make_loader(cache224, self.s["labels"], idx, scfg, False, self.device)
            outs.append(grade_probs(predict_logits(m2, ld, self.device, scfg.amp)[0]))
        record(f"Small CNN from scratch ({c.baseline_img_size}px)", outs[0], outs[1])
        out[f"Small CNN from scratch ({c.baseline_img_size}px)"]["epochs_completed"] = summ["epochs_completed"]
        # primary (same samples, same metric code path)
        record(f"EfficientNet-B0 transfer learning ({c.img_size}px)", self.s["pv_c"], self.s["pt_c"])
        self.metrics["baselines"] = out
        self.s["baseline_probs"] = store
        # paired bootstrap of primary minus each baseline
        paired = {}
        pa = store[f"EfficientNet-B0 transfer learning ({c.img_size}px)"]
        for name, (pb, tb) in store.items():
            if name.startswith("EfficientNet"):
                continue
            paired[name] = {m: M.paired_bootstrap_diff(self.s["yt"], pa[0], pb, m, min(c.n_boot, 500), c.seed, pa[1], tb) for m in ("roc_auc", "qwk")}
        self.metrics["paired_comparison_primary_minus_baseline"] = paired

    # ----------------------------------------------------------------------------- quality gate
    def _infer_raw(self, imgs: list[np.ndarray], enhance_flags: list[bool]) -> np.ndarray:
        """Full inference path from raw RGB arrays (optional CLAHE) -> calibrated 5-grade probabilities."""
        tcfg, model = self.s["tcfg"], self.s["model"]
        xs = []
        for im, e in zip(imgs, enhance_flags):
            _, _, x = preprocess_for_model(enhance_image(im) if e else im, tcfg.img_size, tcfg.use_ben_graham)
            xs.append(x)
        out = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(xs), 16):
                xb = torch.from_numpy(np.stack(xs[i:i + 16])).to(self.device)
                with torch.autocast(device_type="cuda", enabled=tcfg.amp and self.device == "cuda"):
                    out.append(model(xb).float().cpu().numpy())
        return grade_probs(np.concatenate(out), self.s["T"])

    def s_quality(self):
        c, df = self.cfg, self.s["df"]
        rng = np.random.default_rng(c.seed)
        tr = df[df["split"] == "train"]
        sample = tr.sample(min(len(tr), 800), random_state=c.seed)
        thr = fit_thresholds([compute_quality_metrics(read_image(p)) for p in sample["path"]])
        self.s["qthr"] = thr
        cal_path = Path(self.s["tcfg"].out_dir) / self.s["tcfg"].run_name / "calibration.json"
        cal = json.loads(cal_path.read_text())
        cal["quality_thresholds"] = thr.to_dict()
        cal_path.write_text(json.dumps(cal, indent=2))
        te = df[df["split"] == "test"]
        raw = [read_image(p) for p in te["path"]]
        st = [assess_quality(im, thr) for im in raw]
        status = np.array([s.status for s in st])
        clean = {"n": int(len(st)), **{k: int((status == k).sum()) for k in ("good", "enhance", "reject")}}
        # pipeline probabilities (CLAHE applied for 'enhance'); rejected images are still scored so they can be analysed
        probs = self._infer_raw(raw, [s == "enhance" for s in status])
        pref = M.referable_prob(probs)
        yt = self.s["yt"]
        thr_v, margin = self.s["thr"], self.s["margin"]
        pm = {"gate_conservative_review_off": pipeline_metrics(yt, pref, status, thr_v, 0.0),
              "gate_plus_review_band": pipeline_metrics(yt, pref, status, thr_v, margin)}
        rej_idx = np.where(status == "reject")[0]
        rej_examples = [{"image_id": te["image_id"].iloc[i], "true_grade": int(yt[i]), "reason": st[i].reason} for i in rej_idx[:20]]
        # degradations on a stratified test subset
        sub = te.groupby("grade", group_keys=False).apply(lambda g: g.sample(min(len(g), max(3, c.quality_eval_n // 5)), random_state=c.seed))
        sub_pos = [te.index.get_loc(i) for i in sub.index]
        deg_rows, perf_rows = [], []
        for kind in DEGRADATIONS:
            for sev in (1, 2, 3):
                sts = [assess_quality(degrade(raw[j], kind, sev, c.seed), thr).status for j in sub_pos]
                n = len(sts)
                deg_rows.append({"kind": kind, "severity": sev, "n": n, **{k: sts.count(k) / n for k in ("reject", "enhance", "good")}})
        ysub = yt[sub_pos]
        sub_clean_status = status[sub_pos]
        for name, imgs, stat in [("clean (same subset)", [raw[j] for j in sub_pos], list(sub_clean_status))] + \
                [(f"{k} (severity 2)", [degrade(raw[j], k, 2, c.seed) for j in sub_pos], None) for k in DEGRADATIONS]:
            if stat is None:
                stat = [assess_quality(im, thr).status for im in imgs]
            stat = np.array(stat)
            pr = M.referable_prob(self._infer_raw(imgs, [s == "enhance" for s in stat]))
            yb = M.referable_labels(ysub)
            ungated = M.binary_metrics(yb, pr, thr_v)
            keep = stat != "reject"
            gated = M.binary_metrics(yb[keep], pr[keep], thr_v) if keep.any() else None
            cons = pipeline_metrics(ysub, pr, stat, thr_v, 0.0)["conservative"]
            perf_rows.append({"condition": name, "n": int(len(yb)), "rejected_share": float((~keep).mean()), "ungated_sensitivity": ungated["sensitivity"],
                              "ungated_specificity": ungated["specificity"], "ungated_auc": ungated["roc_auc"],
                              "gated_analysed_sensitivity": gated["sensitivity"] if gated else float("nan"),
                              "gated_analysed_specificity": gated["specificity"] if gated else float("nan"),
                              "conservative_sensitivity": cons["sensitivity"], "conservative_specificity": cons["specificity"]})
        self.metrics["quality"] = {"method": "heuristic rules (Laplacian sharpness, illumination, contrast, FOV); NOT a model trained on quality labels",
                                   "thresholds": thr.to_dict(), "clean": clean, "degradation": deg_rows, "model_under_degradation": perf_rows,
                                   "rejected_examples": rej_examples, "pipeline_on_clean_test": pm,
                                   "note": "degradation detection rates measure sensitivity to SYNTHETIC defects only; there are no genuine quality labels (e.g. EyeQ) here."}
        self.s.update(q_status=status, q_pref=pref, q_probs=probs)
        P.plot_quality_degradation(deg_rows, clean, self.fig / "quality_degradation.png")
        j0 = sub_pos[0]
        imgs = [raw[j0]] + [degrade(raw[j0], k, 2, c.seed) for k in DEGRADATIONS]
        P.plot_image_grid([cv2.resize(i, (256, 256)) for i in imgs],
                          ["original"] + [f"{k} (sev 2)\n{assess_quality(degrade(raw[j0], k, 2, c.seed), thr).status}" for k in DEGRADATIONS],
                          self.fig / "quality_examples.png", ncols=7, suptitle="Synthetic degradations and the gate's verdict", figw=1.9)

    # ----------------------------------------------------------------------------- grad-cam
    def s_gradcam(self):
        df, model, tcfg = self.s["df"], self.s["model"], self.s["tcfg"]
        dt, yt, pc = self.s["dt"], self.s["yt"], self.s["pt_c"]
        pred = pc.argmax(1)
        picks, tags = [], []
        for g in range(5):
            ok = np.where((yt == g) & (pred == g))[0]
            bad = np.where((yt == g) & (pred != g))[0]
            for i in ok[:1]:
                picks.append(i); tags.append("correct")
            for i in bad[:1]:
                picks.append(i); tags.append("wrong")
        if not picks:
            raise RuntimeError("no test samples to visualise")
        gc = GradCAM(model)
        ims, titles, cams = [], [], {}
        for i, tag in zip(picks, tags):
            plain = np.asarray(self.s["cache"][int(dt["cache_idx"].iloc[i])])
            x = torch.from_numpy(to_tensor_array(to_model_input(plain, tcfg.use_ben_graham)))[None].to(self.device)
            cam, _ = gc(x, "pred")
            cams[i] = cam
            ims.append(overlay_heatmap(plain, cam))
            titles.append(f"{tag}: true {yt[i]} / pred {pred[i]}\nP(ref)={M.referable_prob(pc[i:i+1])[0]:.2f}")
        gc.remove()
        P.plot_image_grid(ims, titles, self.fig / "gradcam_examples.png", ncols=5, figw=2.3,
                          suptitle="Grad-CAM overlays (model attention, not lesion detection): correct and incorrect predictions by true grade")
        # false negatives / false positives
        pr = M.referable_prob(pc)
        yb = M.referable_labels(yt)
        thr = self.s["thr"]
        fn = np.where((yb == 1) & (pr < thr))[0]
        fp = np.where((yb == 0) & (pr >= thr))[0]
        fn = fn[np.argsort(pr[fn])][:5]
        fp = fp[np.argsort(-pr[fp])][:5]
        gc = GradCAM(model)
        ims, titles = [], []
        for idx, tag in [(i, "FALSE NEG") for i in fn] + [(i, "FALSE POS") for i in fp]:
            plain = np.asarray(self.s["cache"][int(dt["cache_idx"].iloc[idx])])
            x = torch.from_numpy(to_tensor_array(to_model_input(plain, tcfg.use_ben_graham)))[None].to(self.device)
            cam, _ = gc(x, "pred")
            ims.append(overlay_heatmap(plain, cam))
            titles.append(f"{tag}: true {yt[idx]} / pred {pred[idx]}\nP(ref)={pr[idx]:.2f}")
        gc.remove()
        if ims:
            P.plot_image_grid(ims, titles, self.fig / "error_examples.png", ncols=5, figw=2.3, suptitle="Most confident errors at the operating threshold (Grad-CAM overlay)")
        # sanity check: weight randomisation
        rho = []
        for i in picks[:3]:
            plain = np.asarray(self.s["cache"][int(dt["cache_idx"].iloc[i])])
            x = torch.from_numpy(to_tensor_array(to_model_input(plain, tcfg.use_ben_graham)))[None].to(self.device)
            rho.append(randomization_sanity_check(model, x, "pred")["spearman"])
        self.metrics["gradcam"] = {"n_visualised": len(picks), "randomization_spearman_per_image": rho,
                                   "randomization_spearman_mean": float(np.nanmean(rho)),
                                   "note": "Spearman between trained-model and weight-randomised Grad-CAM maps on 3 images; values near 0 mean the map depends on learned weights. "
                                           "Not a proof of lesion localisation.",
                                   "n_false_negatives_listed": int(len(fn)), "n_false_positives_listed": int(len(fp))}

    # ----------------------------------------------------------------------------- latency
    def s_latency(self):
        model, tcfg = self.s["model"], self.s["tcfg"]
        sz = tcfg.img_size
        out = {"device": self.metrics["environment"].get("gpu", "cpu"), "image_size": sz, "precision": "AMP fp16" if (self.device == "cuda" and tcfg.amp) else "fp32"}
        for bs in (1, 16):
            x = torch.randn(bs, 3, sz, sz, device=self.device)
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=tcfg.amp and self.device == "cuda"):
                for _ in range(3):
                    model(x)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                t = time.time()
                reps = 20
                for _ in range(reps):
                    model(x)
                if self.device == "cuda":
                    torch.cuda.synchronize()
            ms = (time.time() - t) / reps * 1000
            out[f"model_forward_ms_per_batch_bs{bs}"] = round(ms, 2)
            out[f"model_forward_ms_per_image_bs{bs}"] = round(ms / bs, 2)
        raw = read_image(self.s["df"]["path"].iloc[0])
        t = time.time()
        for _ in range(5):
            preprocess_for_model(raw, sz, tcfg.use_ben_graham)
            assess_quality(raw)
        out["preprocess_plus_quality_ms_per_image_cpu"] = round((time.time() - t) / 5 * 1000, 1)
        out["note"] = "forward pass on random tensors (timing only); preprocessing timed on one real image; excludes disk read and Grad-CAM."
        self.metrics["latency"] = out

    # ----------------------------------------------------------------------------- external validation
    def s_external(self):
        c = self.cfg
        edf = D.load_labelled_dataset(c.external_root, "external")
        if len(edf) < 20:
            raise RuntimeError(f"only {len(edf)} labelled external images found; skipped")
        raw = [read_image(p) for p in edf["path"]]
        probs = self._infer_raw(raw, [False] * len(raw))
        y = edf["grade"].to_numpy()
        self.metrics["external"] = M.evaluate_predictions(y, probs, self.s["thr"], f"external: {Path(c.external_root).name}",
                                                          "EXTERNAL (never used for training/selection); threshold+temperature from APTOS validation", c.n_boot, c.seed)
        self.metrics["external"]["dataset"] = f"external: {Path(c.external_root).name}"
        self.metrics["external"]["note"] = "Domain shift (different cameras/graders/label protocol) applies; compare with caution."
        P.plot_roc_pr(y, probs, self.s["thr"], self.fig / "external_roc_pr.png", "external")

    # ----------------------------------------------------------------------------- ablation
    def s_ablation(self):
        yt, thr = self.s["yt"], self.s["thr"]
        rows = []
        for name, v in self.metrics.get("baselines", {}).items():
            rows.append({"name": name, "n": v["n"], "sensitivity": v["sensitivity"], "specificity": v["specificity"], "roc_auc": v["roc_auc"], "qwk": v["qwk"],
                         "coverage": "all", "notes": "threshold from validation (sens >= %.2f rule)" % self.cfg.target_sensitivity})
        if "baselines" not in self.metrics:
            h = self.metrics["internal_holdout"]
            rows.append({"name": "EfficientNet-B0 transfer learning", "n": h["n_samples"], **{k: h["screening"][k] for k in ("sensitivity", "specificity", "roc_auc")},
                         "qwk": h["grades"]["qwk"], "coverage": "all", "notes": ""})
        q = self.metrics.get("quality", {}).get("pipeline_on_clean_test")
        if q:
            for key, label in (("gate_conservative_review_off", "EfficientNet-B0 + quality gate (rejected counted as referred)"),
                               ("gate_plus_review_band", "EfficientNet-B0 + quality gate + human-review band (both counted as referred)")):
                m = q[key]
                cons = m["conservative"]
                rows.append({"name": label, "n": m["n"], "sensitivity": cons["sensitivity"], "specificity": cons["specificity"], "roc_auc": float("nan"), "qwk": float("nan"),
                             "coverage": f"{m['coverage_auto']:.1%} auto ({m['n_recapture']} recapture, {m['n_human_review']} review)",
                             "notes": "conservative accounting; AUC/QWK not defined for the decision pipeline"})
        self.metrics["ablation"] = rows
        self.metrics["ablation_notes"] = ("All rows are evaluated on the identical internal test split. Lesion-evidence and lesion/CNN fusion rows are absent because no lesion "
                                          "detector was implemented (no lesion annotations in APTOS). 'Fusion' here means quality gate + confidence-based review band only.")
        ok = [r for r in rows if not (r["name"].startswith("EfficientNet-B0 +"))]
        if ok:
            P.plot_bars([r["name"].replace(" (", "\n(") for r in ok], [r["roc_auc"] for r in ok], self.fig / "ablation_auc.png",
                        "Referable-DR ROC-AUC by model (same test images)", "ROC-AUC", ylim=(0, 1.1))
