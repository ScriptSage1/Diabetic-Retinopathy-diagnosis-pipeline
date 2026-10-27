"""Remote lesion-evidence experiment (Kaggle GPU): IDRiD lesion U-Net + OD/fovea locator -> lesion features for APTOS and IDRiD.

Stages: idrid index -> lesion U-Net (train/val on IDRiD train, thresholds from val, REPORT on IDRiD official test) ->
OD/fovea locator (IDRiD localization) -> lesion features for every APTOS image and every IDRiD grading image ->
CNN probabilities on IDRiD grading images (if the classifier checkpoint is attached) -> figures.

The rule engine / fusion evaluation runs afterwards on CPU from the saved CSVs (scripts/evaluate_lesion_fusion.py).
No lesion-detection performance is ever claimed for APTOS: only the IDRiD test split measures that.
"""
from __future__ import annotations

import json
import math
import platform
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import lesion as L
from . import metrics as M
from . import plots as P
from .datasets import dhash
from .preprocess import read_image, to_tensor_array


@dataclass
class LesionRunConfig:
    idrid_root: str | None = None
    aptos_root: str | None = None
    cnn_checkpoint: str | None = None
    split_manifest: str | None = None
    work_dir: str = "outputs_lesion"
    smoke: bool = False
    iters: int = 4000
    batch: int = 8
    patch: int = 512
    lr: float = 5e-4
    val_n: int = 10
    eval_every: int = 250
    loc_epochs: int = 40
    loc_size: int = 512
    num_workers: int = 4
    seed: int = 42
    pretrained: bool = True
    pretrained_file: str | None = None
    max_minutes: float | None = None
    aptos_limit: int | None = None
    lesion_model_dir: str | None = None   # only_idrid_features mode: directory with lesion_unet.pt, locator.pt, lesion_config.json
    only_idrid_features: bool = False
    eophtha_root: str | None = None   # v2: additional partially-annotated data (MA / EX masks, healthy images)
    v2: bool = False                  # v2 training: pooled data, partial labels, acquisition augmentation, large-region filter
    only_features: bool = False  # re-extract APTOS + IDRiD features with already-trained lesion models


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def attach_labels(rows: list[dict], df: pd.DataFrame) -> pd.DataFrame:
    """Attach grade / dme / split / duplicate flag to extracted feature rows BY POSITION (rows are in df order).
    IDRiD's training and testing sets both number their images from IDRiD_001, so merging on the image name alone would
    cross-join the two sets and mix up labels (this bug was found in the first lesion run)."""
    if len(rows) != len(df):
        raise ValueError(f"{len(rows)} feature rows for {len(df)} images")
    feats = pd.DataFrame(rows)
    if not (feats["image_id"].to_numpy() == df["image_id"].to_numpy()).all():
        raise ValueError("feature rows are not in the same order as the label table")
    for col in ("grade", "dme", "split", "in_lesion_train_or_val"):
        feats[col] = df[col].to_numpy()
    return feats


class LesionRun:
    def __init__(self, cfg: LesionRunConfig):
        self.cfg = cfg
        self.out = Path(cfg.work_dir)
        self.fig, self.res, self.art = self.out / "figures", self.out / "results", self.out / "artifacts" / "lesion"
        for d in (self.fig, self.res, self.art):
            d.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics: dict = {"config": asdict(cfg)}
        self.status: dict = {}
        self.s: dict = {}
        self.t0 = time.time()
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    def stage(self, name, fn, critical=False):
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
        M.save_json({**self.metrics, "stages": self.status}, self.res / "metrics_lesion.json")
        M.save_json(self.status, self.res / "run_status_lesion.json")

    def minutes_left(self):
        return None if not self.cfg.max_minutes else self.cfg.max_minutes - (time.time() - self.t0) / 60

    def run_all(self):
        self.stage("environment", self.s_env)
        self.stage("idrid_index", self.s_index, critical=True)
        self.stage("lesion_unet", self.s_unet_v2 if self.cfg.v2 else self.s_unet, critical=True)
        self.stage("lesion_eval_idrid_test", self.s_eval_seg, critical=True)
        if self.cfg.v2 and self.cfg.eophtha_root:
            self.stage("lesion_eval_eophtha_test", self.s_eval_eophtha)
        self.stage("locator", self.s_locator, critical=True)
        self.stage("save_models", self.s_save, critical=True)
        self.stage("features_idrid_grading", self.s_feat_idrid)
        self.stage("features_aptos", self.s_feat_aptos)
        self.stage("figures", self.s_figs)
        self.metrics["total_minutes"] = round((time.time() - self.t0) / 60, 1)
        self.save()
        log(f"lesion pipeline finished in {self.metrics['total_minutes']} min")

    # ------------------------------------------------------------------ stages
    def s_env(self):
        env = {"python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU run)"}
        self.metrics["environment"] = env
        log(env)

    def s_index(self):
        idx = L.index_idrid(self.cfg.idrid_root)
        self.s["idx"] = idx
        n = {"seg_train": len(idx["seg"]["train"]), "seg_test": len(idx["seg"]["test"]),
             "grading_train": len(idx["grading"].get("train", [])), "grading_test": len(idx["grading"].get("test", [])),
             "loc_train": len(idx["loc"].get("train", [])), "loc_test": len(idx["loc"].get("test", []))}
        log("IDRiD counts:", n)
        if n["seg_train"] == 0 or n["seg_test"] == 0:
            raise RuntimeError(f"IDRiD segmentation images not found under {self.cfg.idrid_root}: {n}")
        lesion_present = {k: sum(1 for it in idx["seg"]["train"].values() if it.get(k)) for k in L.LESIONS}
        self.metrics["idrid"] = {"counts": n, "train_images_with_mask": lesion_present,
                                 "license": "CC BY 4.0 (Porwal et al., IDRiD, Data 2018;3(3):25); this run used a third-party Kaggle copy of the official release",
                                 "note": "A missing mask file is interpreted as 'no lesion of that type in the image' (dataset convention)."}

    def _load_items(self, split: str, limit: int | None = None):
        items = sorted(self.s["idx"]["seg"][split].items())
        if limit:
            items = items[:limit]
        out = []
        for stem, it in items:
            canvas, masks, fov = L.load_seg_item(it)
            out.append({"stem": stem, "canvas": canvas, "masks": masks, "fov": fov})
        return out

    def s_unet(self):
        c = self.cfg
        lim = 8 if c.smoke else None
        tr_all = self._load_items("train", lim)
        rng = np.random.default_rng(c.seed)
        order = rng.permutation(len(tr_all))
        n_val = max(2, min(c.val_n, len(tr_all) // 4)) if not c.smoke else 2
        val = [tr_all[i] for i in order[:n_val]]
        tr = [tr_all[i] for i in order[n_val:]]
        self.s["val_items"], self.s["train_stems"], self.s["val_stems"] = val, [t["stem"] for t in tr], [v["stem"] for v in val]
        log(f"lesion U-Net: {len(tr)} train / {len(val)} val images (IDRiD segmentation training set)")
        model = L.LesionUNet(pretrained=c.pretrained, pretrained_file=c.pretrained_file).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, c.iters)))))
        scaler = torch.amp.GradScaler("cuda", enabled=self.device == "cuda")
        imgs, masks = [t["canvas"] for t in tr], [t["masks"] for t in tr]
        iters = 20 if c.smoke else c.iters
        best, best_it, hist = -1.0, 0, []
        run_loss, t_start = 0.0, time.time()
        stop_reason = "completed all iterations"
        for it in range(1, iters + 1):
            model.train()
            x, y = L.sample_batch(imgs, masks, c.batch if not c.smoke else 2, c.patch if not c.smoke else 256, rng)
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                logits = model(x)
            loss = L.seg_loss(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at iteration {it}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            if it % (c.eval_every if not c.smoke else 10) == 0 or it == iters:
                st = self.eval_items(model, val)
                aupr = [st.aupr(L.LESIONS.index(k)) for k in L.FINDINGS]
                score = float(np.nanmean(aupr)) if not np.all(np.isnan(aupr)) else 0.0
                hist.append({"iteration": it, "train_loss": run_loss / (c.eval_every if not c.smoke else 10), "val_mean_aupr": score,
                             **{f"val_aupr_{k}": a for k, a in zip(L.FINDINGS, aupr)}, "minutes": (time.time() - t_start) / 60})
                run_loss = 0.0
                log(f"it {it}/{iters} loss {hist[-1]['train_loss']:.4f} val mean AUPR {score:.4f} " + " ".join(f"{k}:{a:.3f}" for k, a in zip(L.FINDINGS, aupr)))
                torch.save({"model_state": model.state_dict(), "iteration": it}, self.art / "lesion_unet_last.pt")
                if score > best:
                    best, best_it = score, it
                    torch.save({"model_state": model.state_dict(), "iteration": it, "val_mean_aupr": score}, self.art / "lesion_unet.pt")
                ml = self.minutes_left()
                if ml is not None and ml < 5:
                    stop_reason = "stopped: time budget"
                    break
        pd.DataFrame(hist).to_csv(self.res / "lesion_training_history.csv", index=False)
        ck = torch.load(self.art / "lesion_unet.pt", map_location=self.device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        self.s["seg"] = model
        self.metrics["lesion_training"] = {"iterations_run": it, "best_iteration": best_it, "best_val_mean_aupr": best, "stop_reason": stop_reason,
                                          "minutes": round((time.time() - t_start) / 60, 1), "n_train_images": len(tr), "n_val_images": len(val),
                                          "batch": c.batch, "patch": c.patch, "canvas": L.CANVAS}
        # thresholds from VALIDATION images only
        stv = self.eval_items(model, val)
        thr = {}
        for k in L.LESIONS:
            t, f1 = stv.best_threshold(L.LESIONS.index(k))
            thr[k] = float(np.clip(t, 0.05, 0.95)) if not math.isnan(f1) else 0.5
        self.s["thr"] = thr
        self.metrics["lesion_thresholds_fitted_on_validation"] = thr

    # ------------------------------------------------------------------ v2 training
    def s_unet_v2(self):
        """Pooled training on IDRiD (all 5 classes annotated) + e-ophtha (only MA/EX, per-image known channels) with a prefetching sampler."""
        import queue
        import threading

        c = self.cfg
        smoke = c.smoke
        idr = sorted(self.s["idx"]["seg"]["train"].items())[: 8 if smoke else None]
        rng = np.random.default_rng(c.seed)
        order = rng.permutation(len(idr))
        n_val = 2 if smoke else max(2, min(c.val_n, len(idr) // 4))
        val_idr = [idr[i] for i in order[:n_val]]
        tr_idr = [idr[i] for i in order[n_val:]]
        log(f"v2: IDRiD {len(tr_idr)} train / {len(val_idr)} val images")
        plains, masks, valids, group = [], [], [], []
        for stem, it in tr_idr:
            p, m, v, _ = L.load_plain_idrid(it)
            plains.append(p), masks.append(m), valids.append(v), group.append("idrid")
        val_sets = {"idrid": [(stem, *L.load_plain_idrid(it)[:3]) for stem, it in val_idr]}
        eo = {"train": [], "val": [], "test": []}
        if c.eophtha_root:
            items = L.index_eophtha(c.eophtha_root)
            if smoke:
                items = [i for i in items if i["masks"]["MA"] or i["masks"]["EX"]][:6] + [i for i in items if i["healthy"]][:3]
            eo = L.split_by_patient(items, c.seed)
            log(f"v2: e-ophtha {len(eo['train'])} train / {len(eo['val'])} val / {len(eo['test'])} test items (patient-level split)")
            for it in eo["train"]:
                p, m, v, _ = L.load_plain_eophtha(it)
                plains.append(p), masks.append(m), valids.append(v), group.append("eophtha_healthy" if it["healthy"] else "eophtha")
            val_sets["eophtha"] = [(it["id"], *L.load_plain_eophtha(it)[:3]) for it in eo["val"]]
            self.s["eo_test_items"] = eo["test"]
        share = {"idrid": 0.35, "eophtha": 0.40, "eophtha_healthy": 0.25}
        cnt = {g: max(1, group.count(g)) for g in share}
        w = np.array([share[g] / cnt[g] if group.count(g) else 0.0 for g in group])
        w = w / w.sum()
        self.s["max_area"] = L.component_area_limits(masks, valids)
        log("max region area limits (canvas px):", self.s["max_area"])
        model = L.LesionUNet(pretrained=c.pretrained, pretrained_file=c.pretrained_file).to(self.device)
        iters = 20 if smoke else c.iters
        batch, patch = (2, 256) if smoke else (c.batch, c.patch)
        opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, iters)))))
        scaler = torch.amp.GradScaler("cuda", enabled=self.device == "cuda")
        q: "queue.Queue" = queue.Queue(maxsize=6)
        stop = threading.Event()

        def producer(seed):
            r = np.random.default_rng(seed)
            while not stop.is_set():
                try:
                    q.put(L.sample_batch_v2(plains, masks, valids, w, batch, patch, r), timeout=1)
                except queue.Full:
                    continue

        threads = [threading.Thread(target=producer, args=(c.seed + 1 + i,), daemon=True) for i in range(max(1, min(3, c.num_workers)))]
        [t.start() for t in threads]

        def val_score(model_):
            vals = []
            sts = {}
            for name, sets in val_sets.items():
                st = L.PixelStats()
                for _, plain, mk, vd in sets:
                    chans = [k for k in range(len(L.LESIONS)) if vd[k]]
                    st.add(L.predict_probs(model_, L.bg_from_plain(plain), self.device), mk, valid=L.circle_mask(L.CANVAS, 0.98), channels=chans)
                sts[name] = st
                for k in L.FINDINGS:
                    a = st.aupr(L.LESIONS.index(k))
                    if not math.isnan(a):
                        vals.append(a)
            return (float(np.mean(vals)) if vals else 0.0), sts

        best, best_it, hist, run_loss, t_start = -1.0, 0, [], 0.0, time.time()
        stop_reason = "completed all iterations"
        try:
            for it in range(1, iters + 1):
                model.train()
                x, y, v = q.get(timeout=300)
                x, y, v = x.to(self.device), y.to(self.device), v.to(self.device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                    logits = model(x)
                loss = L.seg_loss(logits.float(), y, v)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at iteration {it}")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                run_loss += loss.item()
                every = 10 if smoke else c.eval_every
                if it % every == 0 or it == iters:
                    score, _ = val_score(model)
                    hist.append({"iteration": it, "train_loss": run_loss / every, "val_mean_aupr": score, "minutes": (time.time() - t_start) / 60})
                    run_loss = 0.0
                    log(f"it {it}/{iters} loss {hist[-1]['train_loss']:.4f} val mean AUPR (IDRiD + e-ophtha) {score:.4f}")
                    torch.save({"model_state": model.state_dict(), "iteration": it}, self.art / "lesion_unet_last.pt")
                    if score > best:
                        best, best_it = score, it
                        torch.save({"model_state": model.state_dict(), "iteration": it, "val_mean_aupr": score}, self.art / "lesion_unet.pt")
                    ml = self.minutes_left()
                    if ml is not None and ml < 5:
                        stop_reason = "stopped: time budget"
                        break
        finally:
            stop.set()
        pd.DataFrame(hist).to_csv(self.res / "lesion_training_history.csv", index=False)
        ck = torch.load(self.art / "lesion_unet.pt", map_location=self.device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        self.s["seg"] = model
        self.s["val_items"] = [{"masks": mk} for _, _, mk, _ in val_sets["idrid"]]  # used by the locator stage to measure disc diameters
        _, sts = val_score(model)
        thr = {}
        for k in L.LESIONS:
            i = L.LESIONS.index(k)
            pooled = L.PixelStats()
            for st in sts.values():
                pooled.pos[i] += st.pos[i]
                pooled.neg[i] += st.neg[i]
            t, f1 = pooled.best_threshold(i)
            thr[k] = float(np.clip(t, 0.05, 0.95)) if not math.isnan(f1) else 0.5
        self.s["thr"] = thr
        self.metrics["lesion_training"] = {"version": "v2 (IDRiD + e-ophtha, partial labels, acquisition augmentation)", "iterations_run": it, "best_iteration": best_it,
                                          "best_val_mean_aupr": best, "stop_reason": stop_reason, "minutes": round((time.time() - t_start) / 60, 1),
                                          "n_train_images": len(plains), "n_train_by_source": {g: group.count(g) for g in share}, "n_val_idrid": len(val_idr),
                                          "n_val_eophtha": len(val_sets.get("eophtha", [])), "batch": batch, "patch": patch, "canvas": L.CANVAS, "max_area": self.s["max_area"]}
        self.metrics["lesion_thresholds_fitted_on_validation"] = thr

    def s_eval_eophtha(self):
        """Held-out e-ophtha PATIENTS: AUPR / Dice / IoU for MA and EX (the annotated channels) with the validation thresholds."""
        items = self.s.get("eo_test_items") or []
        st = L.PixelStats()
        n = {"MA": 0, "EX": 0}
        for it in items:
            plain, mk, vd, _ = L.load_plain_eophtha(it)
            st.add(L.predict_probs(self.s["seg"], L.bg_from_plain(plain), self.device), mk, valid=L.circle_mask(L.CANVAS, 0.98), channels=[k for k in range(len(L.LESIONS)) if vd[k]])
            for k in n:
                n[k] += int(vd[L.LESIONS.index(k)] and mk[L.LESIONS.index(k)].any())
        per = {k: {"aupr": st.aupr(L.LESIONS.index(k)), "n_test_images_with_lesion": n[k], **st.at_threshold(L.LESIONS.index(k), self.s["thr"][k])} for k in ("MA", "EX")}
        self.metrics["lesion_segmentation_eophtha_test"] = {"dataset": f"e-ophtha held-out patients ({len(items)} images incl. healthy), different clinics/cameras than IDRiD",
                                                            "per_lesion": per, "note": "only MA and EX are annotated in e-ophtha; thresholds chosen on validation images"}
        log({k: {m: round(v, 3) for m, v in d.items() if isinstance(v, float) and not math.isnan(v)} for k, d in per.items()})

    def eval_items(self, model, items) -> L.PixelStats:
        st = L.PixelStats()
        for it in items:
            probs = L.predict_probs(model, it["canvas"], self.device)
            st.add(probs, it["masks"], valid=L.circle_mask(L.CANVAS, 0.98))
        return st

    def s_eval_seg(self):
        test = self._load_items("test", 6 if self.cfg.smoke else None)
        self.s["test_items"] = test
        st = self.eval_items(self.s["seg"], test)
        per = {}
        for k in L.LESIONS:
            i = L.LESIONS.index(k)
            per[k] = {"aupr": st.aupr(i), "n_test_images_with_lesion": int(sum(1 for t in test if t["masks"][i].any())),
                      **st.at_threshold(i, self.s["thr"][k])}
        self.metrics["lesion_segmentation_idrid_test"] = {
            "dataset": f"IDRiD official segmentation test split ({len(test)} images)", "evaluation_type": "held-out test of the lesion model (images never used for training or selection)",
            "per_lesion": per, "metric_notes": "pixel-level on the 1280-px retina canvas; Dice == pixel F1; thresholds chosen on validation images",
            "canvas_px": L.CANVAS}
        log({k: {m: round(v, 3) for m, v in d.items() if isinstance(v, float) and not math.isnan(v)} for k, d in per.items()})
        # PR curves figure
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.2))
        for k in L.FINDINGS:
            prec, rec, _, _, Pn = st.curves(L.LESIONS.index(k))
            if Pn:
                ax.plot(rec, prec, label=f"{k} (AUPR {per[k]['aupr']:.2f})")
        ax.set(xlabel="Pixel recall", ylabel="Pixel precision", title="Lesion segmentation: precision-recall on IDRiD test", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.fig / "lesion_pr_idrid_test.png", dpi=160)
        plt.close(fig)
        # qualitative panels: image | ground truth | prediction
        ims, titles = [], []
        for t in test[:4]:
            probs = L.predict_probs(self.s["seg"], t["canvas"], self.device)
            base = t["canvas"].copy()
            gt, pr = base.copy(), base.copy()
            for k in ("EX", "SE", "HE", "MA"):
                i = L.LESIONS.index(k)
                col = np.array(L.LESION_COLORS[k])
                mg = t["masks"][i] > 0
                mp = probs[i].astype(np.float32) > self.s["thr"][k]
                if k == "MA":
                    mg = cv2.dilate(mg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    mp = cv2.dilate(mp.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                gt[mg], pr[mp] = col, col
            small = lambda a: cv2.resize(a, (480, 480), interpolation=cv2.INTER_AREA)
            plain = cv2.resize(t["fov"].to_canvas(read_image(self.s["idx"]["seg"]["test"][t["stem"]]["image"])), (480, 480), interpolation=cv2.INTER_AREA)
            ims += [plain, small(gt), small(pr)]
            titles += [f"{t['stem']} photograph", "ground truth", "prediction"]
        P.plot_image_grid(ims, titles, self.fig / "lesion_idrid_test_examples.png", ncols=3, figw=3.4,
                          suptitle="IDRiD test images: red=MA, orange=HE, yellow=EX, cyan=SE (annotations vs model)")

    def s_locator(self):
        c = self.cfg
        idx = self.s["idx"]["loc"]
        if "train" not in idx or "test" not in idx:
            raise RuntimeError(f"IDRiD localization labels not found (have {list(idx)})")
        lim = 24 if c.smoke else None

        def load(df):
            X, Y, F = [], [], []
            for r in df.head(lim).itertuples() if lim else df.itertuples():
                if not r.path:
                    continue
                img = read_image(r.path)
                fov = L.make_fov(img, c.loc_size)
                plain = fov.to_canvas(img)
                plain = cv2.resize(plain, (c.loc_size, c.loc_size), interpolation=cv2.INTER_AREA) if plain.shape[0] != c.loc_size else plain
                pts = fov.pts_to_canvas([[r.od_x, r.od_y], [r.fo_x, r.fo_y]]) / c.loc_size
                X.append(L.ben_graham(plain))
                Y.append(pts.reshape(-1).astype(np.float32))
                F.append((fov, r.image_id))
            return np.stack(X), np.stack(Y), F

        Xtr, Ytr, Ftr = load(idx["train"])
        Xte, Yte, Fte = load(idx["test"])
        n_val = max(4, len(Xtr) // 10)
        rng = np.random.default_rng(c.seed)
        perm = rng.permutation(len(Xtr))
        vi, ti = perm[:n_val], perm[n_val:]
        model = L.PointRegressor(pretrained=c.pretrained, pretrained_file=c.pretrained_file).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
        epochs = 1 if c.smoke else c.loc_epochs
        steps = epochs * math.ceil(len(ti) / 16)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-4, total_steps=max(2, steps))
        scaler = torch.amp.GradScaler("cuda", enabled=self.device == "cuda")

        def run_eval(X, Y):
            model.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 16):
                    xb = torch.from_numpy(np.stack([to_tensor_array(a) for a in X[i : i + 16]])).to(self.device)
                    out.append(model(xb).float().cpu().numpy())
            p = np.concatenate(out)
            return p, float(np.abs(p - Y).mean())

        best = 9.0
        for ep in range(1, epochs + 1):
            model.train()
            rng.shuffle(ti)
            for i in range(0, len(ti), 16):
                b = ti[i : i + 16]
                xs, ys = [], []
                for j in b:
                    im, y = Xtr[j], Ytr[j].copy()
                    if rng.random() < 0.5:
                        im = im[:, ::-1]
                        y[[0, 2]] = 1 - y[[0, 2]]
                    if rng.random() < 0.5:
                        im = im[::-1]
                        y[[1, 3]] = 1 - y[[1, 3]]
                    f = np.clip(im.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.uniform(-8, 8), 0, 255).astype(np.uint8)
                    xs.append(to_tensor_array(np.ascontiguousarray(f)))
                    ys.append(y)
                xb, yb = torch.from_numpy(np.stack(xs)).to(self.device), torch.from_numpy(np.stack(ys)).to(self.device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                    pred = model(xb)
                loss = nn.functional.smooth_l1_loss(pred.float(), yb, beta=0.02)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if sched.last_epoch + 1 < steps:
                    sched.step()
            _, verr = run_eval(Xtr[vi], Ytr[vi])
            if verr < best:
                best = verr
                torch.save({"model_state": model.state_dict(), "epoch": ep, "val_l1": verr}, self.art / "locator.pt")
            if ep % 5 == 0 or ep == epochs:
                log(f"locator epoch {ep}/{epochs} val L1 (fraction of canvas) {verr:.4f}")
        ck = torch.load(self.art / "locator.pt", map_location=self.device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        self.s["loc"] = model
        p, _ = run_eval(Xte, Yte)
        errs = []
        for (fov, iid), pp, yy in zip(Fte, p, Yte):
            pred_o = fov.pts_to_orig(pp.reshape(2, 2) * c.loc_size)
            true_o = fov.pts_to_orig(yy.reshape(2, 2) * c.loc_size)
            errs.append(np.linalg.norm(pred_o - true_o, axis=1))
        errs = np.array(errs)  # [n, 2] original-pixel errors (OD, fovea)
        od_d = np.median([np.sqrt(it["masks"][L.LESIONS.index("OD")].sum() / np.pi) * 2 for it in self.s["val_items"] if it["masks"][-1].any()] or [0.09 * L.CANVAS])
        dd_orig = float(od_d / (L.CANVAS / np.median([f.side for f, _ in Fte])))
        self.metrics["locator_idrid_test"] = {
            "n_test": int(len(errs)), "od_error_px_mean": float(errs[:, 0].mean()), "od_error_px_median": float(np.median(errs[:, 0])),
            "fovea_error_px_mean": float(errs[:, 1].mean()), "fovea_error_px_median": float(np.median(errs[:, 1])),
            "disc_diameter_px_approx": dd_orig, "fovea_within_1DD": float((errs[:, 1] <= dd_orig).mean()), "od_within_1DD": float((errs[:, 0] <= dd_orig).mean()),
            "note": "errors in ORIGINAL IDRiD pixels (4288x2848 images); DD = median optic-disc diameter measured on validation masks",
            "val_l1_fraction_of_canvas": float(best), "epochs": epochs}
        log(self.metrics["locator_idrid_test"])

    def s_save(self):
        c = self.cfg
        cfg = {"thresholds": self.s["thr"], "min_area": L.DEFAULT_MIN_AREA, "max_area": self.s.get("max_area"), "locator_size": c.loc_size, "canvas": L.CANVAS,
               "trained_on": "IDRiD segmentation train (lesions) and localization train (OD/fovea)", "min_area_note": "fixed defaults, canvas pixels"}
        (self.art / "lesion_config.json").write_text(json.dumps(cfg, indent=2))
        (self.art / "lesion_unet_last.pt").unlink(missing_ok=True)
        self.s["analyzer"] = L.LesionAnalyzer(self.art, self.device)

    # ------------------------------------------------------------------ feature extraction
    def _extract(self, paths: list[str], ids: list[str]) -> list[dict]:
        an = self.s["analyzer"]

        class DS(torch.utils.data.Dataset):
            def __len__(self_):
                return len(paths)

            def __getitem__(self_, i):
                try:
                    canvas, fov = L.canvas_for_model(read_image(paths[i]), L.CANVAS)
                    return torch.from_numpy(canvas), torch.from_numpy(fov.as_array()), i, True
                except Exception:
                    return torch.zeros(L.CANVAS, L.CANVAS, 3, dtype=torch.uint8), torch.zeros(6, dtype=torch.int64), i, False

        dl = torch.utils.data.DataLoader(DS(), batch_size=2, num_workers=self.cfg.num_workers, shuffle=False)
        rows, t0 = [], time.time()
        for bi, (canv, fa, ii, ok) in enumerate(dl):
            for k in range(len(ii)):
                i = int(ii[k])
                if not bool(ok[k]):
                    rows.append({"image_id": ids[i], "loc_ok": 0})
                    continue
                out = an.analyze_canvas(canv[k].numpy())
                r = {"image_id": ids[i], **out["features"], "od_x": float(out["od_xy"][0]), "od_y": float(out["od_xy"][1]),
                     "fo_x": float(out["fovea_xy"][0]), "fo_y": float(out["fovea_xy"][1])}
                rows.append(r)
            if bi % 100 == 0:
                log(f"features {len(rows)}/{len(paths)} ({(time.time() - t0) / 60:.1f} min)")
        return rows

    def s_feat_idrid(self):
        g = self.s["idx"]["grading"]
        frames = []
        for split in ("train", "test"):
            if split in g:
                d = g[split].dropna(subset=["path"]).copy()
                d["split"] = split
                frames.append(d)
        df = pd.concat(frames, ignore_index=True)
        if self.cfg.smoke:
            df = df.groupby("split").head(12).reset_index(drop=True)
        # leakage guard: flag grading images that are (near-)duplicates of the lesion-model's training/validation images
        seen = []
        for split in ("train",):
            for stem, it in self.s["idx"]["seg"][split].items():
                seen.append(np.packbits(dhash(read_image(it["image"]))))
        seen = np.unpackbits(np.stack(seen), axis=1)
        flags = []
        for p in df["path"]:
            h = np.unpackbits(np.packbits(dhash(read_image(p))))
            flags.append(int(((seen != h).sum(1) <= 6).any()))
        df["in_lesion_train_or_val"] = flags
        log(f"IDRiD grading images flagged as duplicates of lesion-model training images: {sum(flags)}/{len(df)}")
        rows = self._extract(df["path"].tolist(), df["image_id"].tolist())
        feats = attach_labels(rows, df)
        self._final_locator_error(df, feats)
        feats.to_csv(self.res / "lesion_features_idrid.csv", index=False)
        self.metrics["idrid_features"] = {"n": int(len(feats)), "n_flagged_duplicate_of_lesion_training": int(sum(flags))}
        self._cnn_probs_idrid(df)

    def _cnn_probs_idrid(self, df: pd.DataFrame):
        """Classifier probabilities on IDRiD grading images (external check of the classifier). Needs the classifier checkpoint."""
        if self.cfg.cnn_checkpoint and Path(self.cfg.cnn_checkpoint).is_file():
            from .models import grade_probs, load_checkpoint
            from .preprocess import preprocess_for_model

            cal = Path(self.cfg.cnn_checkpoint).parent / "calibration.json"
            T = json.loads(cal.read_text()).get("temperature", 1.0) if cal.is_file() else 1.0
            model, ck = load_checkpoint(self.cfg.cnn_checkpoint, self.device)
            size, bg = ck["cfg"]["img_size"], ck["cfg"].get("use_ben_graham", True)
            out = []
            with torch.no_grad():
                for p in df["path"]:
                    _, _, x = preprocess_for_model(read_image(p), size, bg)
                    with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                        lg = model(torch.from_numpy(x)[None].to(self.device)).float().cpu().numpy()
                    out.append(grade_probs(lg, T)[0])
            cp = pd.DataFrame(np.stack(out), columns=[f"p_grade{k}" for k in range(5)])
            cp.insert(0, "image_id", df["image_id"].values)
            if "split" in df:
                cp.insert(1, "split", df["split"].values)  # image names repeat across IDRiD train and test
            cp.to_csv(self.res / "idrid_cnn_probs.csv", index=False)
            self.metrics["idrid_cnn"] = {"checkpoint": str(self.cfg.cnn_checkpoint), "temperature": T, "n": len(cp)}
        else:
            self.metrics["idrid_cnn"] = {"note": "classifier checkpoint not attached; IDRiD CNN predictions not computed"}

    def _final_locator_error(self, df: pd.DataFrame, feats: pd.DataFrame):
        """Position error of the FINAL disc/fovea estimate (segmentation disc + consistency-checked regressor / anatomical prior) against the
        IDRiD ground truth centres, in original pixels, on the grading test images (the regressor never saw them)."""
        loc = self.s["idx"]["loc"].get("test")
        if loc is None or "split" not in feats:
            return
        te = feats[(feats["split"] == "test") & (feats["loc_ok"] == 1)]
        gt = loc.set_index("image_id")
        errs, falls = [], []
        for i, r in te.iterrows():
            if r["image_id"] not in gt.index:
                continue
            path = df.loc[(df["split"] == "test") & (df["image_id"] == r["image_id"]), "path"].iloc[0]
            fov = L.make_fov(read_image(path), L.CANVAS)
            est = fov.pts_to_orig([[r["od_x"], r["od_y"]], [r["fo_x"], r["fo_y"]]])
            g = gt.loc[r["image_id"]]
            true = np.array([[g["od_x"], g["od_y"]], [g["fo_x"], g["fo_y"]]], float)
            errs.append(np.linalg.norm(est - true, axis=1))
            falls.append(int(r["loc_fallback"]))
        if not errs:
            return
        errs = np.array(errs)
        dd = float(self.metrics.get("locator_idrid_test", {}).get("disc_diameter_px_approx", 500.0))
        self.metrics["locator_final_idrid_test"] = {
            "n_test": int(len(errs)), "od_error_px_median": float(np.median(errs[:, 0])), "od_error_px_mean": float(errs[:, 0].mean()),
            "fovea_error_px_median": float(np.median(errs[:, 1])), "fovea_error_px_mean": float(errs[:, 1].mean()),
            "od_within_1DD": float((errs[:, 0] <= dd).mean()), "fovea_within_1DD": float((errs[:, 1] <= dd).mean()), "disc_diameter_px_approx": dd,
            "share_using_anatomical_prior": float(np.mean(falls)),
            "note": "final estimate = segmented disc + regressor fovea if consistent, else anatomical prior; errors in original IDRiD pixels"}

    def run_only_features(self):
        """Re-extract features for IDRiD grading and APTOS images with already-trained lesion models (no training)."""
        self.stage("environment", self.s_env)

        def load():
            self.s["idx"] = L.index_idrid(self.cfg.idrid_root)
            self.s["analyzer"] = L.LesionAnalyzer(self.cfg.lesion_model_dir, self.device)

        self.stage("load_models", load, critical=True)
        self.stage("features_idrid_grading", self.s_feat_idrid)
        self.stage("features_aptos", self.s_feat_aptos)
        self.metrics["total_minutes"] = round((time.time() - self.t0) / 60, 1)
        self.save()

    def run_only_idrid_features(self):
        """Re-extract the IDRiD lesion features (and classifier probabilities) with already-trained lesion models."""
        self.stage("environment", self.s_env)

        def go():
            self.s["idx"] = L.index_idrid(self.cfg.idrid_root)
            self.s["analyzer"] = L.LesionAnalyzer(self.cfg.lesion_model_dir, self.device)
            self.s_feat_idrid()

        self.stage("idrid_features", go, critical=True)

    def run_only_cnn(self):
        """Small follow-up job: classifier probabilities on the IDRiD grading TEST images (needs the classifier kernel output attached)."""
        self.stage("environment", self.s_env)

        def go():
            idx = L.index_idrid(self.cfg.idrid_root)
            df = idx["grading"]["test"].dropna(subset=["path"]).copy()
            if self.cfg.smoke:
                df = df.head(12)
            self._cnn_probs_idrid(df)

        self.stage("idrid_cnn", go, critical=True)

    def s_feat_aptos(self):
        from .datasets import load_labelled_dataset

        df = load_labelled_dataset(self.cfg.aptos_root, "APTOS 2019")
        if self.cfg.smoke:
            df = df.groupby("grade").head(8).reset_index(drop=True)
        elif self.cfg.aptos_limit:
            df = df.head(self.cfg.aptos_limit)
        if self.cfg.split_manifest and Path(self.cfg.split_manifest).is_file():
            man = pd.read_csv(self.cfg.split_manifest)[["image_id", "split"]]
            df = df.merge(man, on="image_id", how="left")
        else:
            df["split"] = np.nan
        rows = self._extract(df["path"].tolist(), df["image_id"].tolist())
        feats = pd.DataFrame(rows).merge(df[["image_id", "grade", "split"]], on="image_id")
        feats.to_csv(self.res / "lesion_features_aptos.csv", index=False)
        self.metrics["aptos_features"] = {"n": int(len(feats)), "n_failed": int((feats["loc_ok"].fillna(0) == 0).sum()), "n_with_split": int(feats["split"].notna().sum()),
                                          "note": "lesion models applied to APTOS WITHOUT any APTOS lesion annotation: domain transfer from IDRiD, performance on APTOS lesions is unmeasured"}

    def s_figs(self):
        an = self.s["analyzer"]
        feats = pd.read_csv(self.res / "lesion_features_aptos.csv")
        from .datasets import load_labelled_dataset

        df = load_labelled_dataset(self.cfg.aptos_root, "APTOS 2019").set_index("image_id")
        picks = []
        for g in range(5):
            ids = feats[(feats["grade"] == g) & (feats["loc_ok"] == 1)]["image_id"].tolist()[:2]
            picks += [(i, g) for i in ids]
        ims, titles = [], []
        for iid, g in picks:
            out = an.analyze(read_image(df.loc[iid, "path"]))
            f = out["features"]
            ims.append(cv2.resize(out["overlay"], (420, 420), interpolation=cv2.INTER_AREA))
            titles.append(f"APTOS grade {g}\nMA {f['MA_count']} HE {f['HE_count']} EX {f['EX_count']} SE {f['SE_count']}")
        if ims:
            P.plot_image_grid(ims, titles, self.fig / "lesion_examples_aptos.png", ncols=5, figw=2.8,
                              suptitle="Lesion detections on APTOS (domain transfer from IDRiD, no APTOS lesion annotations): red=MA, orange=HE, yellow=EX, cyan=SE; green cross = estimated fovea")
