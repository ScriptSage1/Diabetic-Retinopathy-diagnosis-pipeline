"""Discrete-event simulation (SimPy) of a DR screening programme.

    arrival -> camera queue -> imaging -> upload (2G/3G delay) -> AI inference -> quality check
            -> (reject: recapture, up to a limit)  -> flagged / uncertain -> reviewer queue -> referral

EVERYTHING numeric here is an ASSUMPTION (see SimConfig), not observed operational data. Results show how the
queueing system behaves under those assumptions; they are not evidence of clinical effectiveness.

Time unit: working minutes. Overnight/weekend gaps are not modelled (queues carry seamlessly from one working
day to the next), which understates waiting time experienced by a patient in calendar terms.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simpy


@dataclass
class SimConfig:
    annual_patients: int = 100_000      # roadmap target programme size (assumption)
    working_days: int = 250
    hours_per_day: float = 8.0
    n_cameras: int = 4
    n_reviewers: int = 2
    n_inference: int = 1
    imaging_min: float = 6.0            # mean minutes per fundus imaging session (assumption)
    imaging_cv: float = 0.35            # coefficient of variation of imaging time
    upload_min: float = 2.0             # mean upload delay on 2G/3G (assumption); not capacity limited
    inference_min: float = 0.25         # mean AI inference + report time per image
    review_min: float = 4.0             # mean reviewer time per case (assumption; roadmap target <30 s only for report reading)
    p_quality_reject: float = 0.08      # share failing the quality gate per capture (assumption)
    max_recaptures: int = 1
    prevalence: float = 0.15            # referable-DR prevalence in the screened population (assumption)
    sensitivity: float = 0.90           # AI sensitivity for referable DR (assumption; replace with measured value)
    specificity: float = 0.85           # AI specificity (assumption; replace with measured value)
    p_uncertain: float = 0.08           # share of otherwise-negative cases routed to review as uncertain
    sim_days: int = 40
    warmup_days: int = 5
    seed: int = 0

    @property
    def day_min(self) -> float:
        return self.hours_per_day * 60

    @property
    def arrival_rate(self) -> float:  # patients per working minute
        return self.annual_patients / self.working_days / self.day_min


def _lognormal(rng, mean, cv):
    s2 = math.log(1 + cv**2)
    return rng.lognormal(math.log(mean) - s2 / 2, math.sqrt(s2))


def run(cfg: SimConfig, service: str = "lognormal") -> dict:
    """Run one scenario. service='exponential' uses exponential times everywhere (for the Little's-law/M/M/c check)."""
    rng = np.random.default_rng(cfg.seed)
    env = simpy.Environment()
    cam = simpy.Resource(env, cfg.n_cameras)
    inf = simpy.Resource(env, cfg.n_inference)
    rev = simpy.Resource(env, cfg.n_reviewers)
    T_end, T_warm = cfg.sim_days * cfg.day_min, cfg.warmup_days * cfg.day_min
    rec = {"cam_wait": [], "rev_wait": [], "total": [], "done_t": [], "referred": 0, "cleared": 0, "missed": 0, "no_review": 0, "recaptures": 0,
           "recapture_failed": 0, "n_arrived": 0}
    in_system = [0]
    samples = {"cam_q": [], "rev_q": [], "cam_busy": [], "rev_busy": [], "inf_busy": [], "in_sys": []}

    def draw(mean, cv=0.35):
        return rng.exponential(mean) if service == "exponential" else _lognormal(rng, mean, cv)

    def patient(t_arr):
        in_system[0] += 1
        counted = t_arr >= T_warm
        attempts = 0
        while True:
            t0 = env.now
            with cam.request() as r:
                yield r
                if counted:
                    rec["cam_wait"].append(env.now - t0)
                yield env.timeout(draw(cfg.imaging_min, cfg.imaging_cv))
            yield env.timeout(draw(cfg.upload_min, 0.6))
            with inf.request() as r:
                yield r
                yield env.timeout(draw(cfg.inference_min, 0.3))
            if rng.random() < cfg.p_quality_reject:
                if attempts < cfg.max_recaptures:
                    attempts += 1
                    if counted:
                        rec["recaptures"] += 1
                    continue
                forced_review, uncertain = True, False
                if counted:
                    rec["recapture_failed"] += 1
            else:
                forced_review = False
            break
        truth = rng.random() < cfg.prevalence
        flagged = (rng.random() < cfg.sensitivity) if truth else (rng.random() > cfg.specificity)
        uncertain = (not flagged) and rng.random() < cfg.p_uncertain
        if flagged or uncertain or forced_review:
            t1 = env.now
            with rev.request() as r:
                yield r
                if counted:
                    rec["rev_wait"].append(env.now - t1)
                yield env.timeout(draw(cfg.review_min, 0.5))
            if counted:
                rec["referred" if truth else "cleared"] += 1
        elif counted:
            rec["missed" if truth else "no_review"] += 1
        if counted:
            rec["total"].append(env.now - t_arr)
            rec["done_t"].append(env.now)
        in_system[0] -= 1

    def arrivals():
        while True:
            yield env.timeout(rng.exponential(1.0 / cfg.arrival_rate))
            if env.now >= T_warm:
                rec["n_arrived"] += 1
            env.process(patient(env.now))

    def monitor():
        while True:
            if env.now >= T_warm:
                samples["cam_q"].append(len(cam.queue))
                samples["rev_q"].append(len(rev.queue))
                samples["cam_busy"].append(cam.count / cfg.n_cameras)
                samples["rev_busy"].append(rev.count / cfg.n_reviewers)
                samples["inf_busy"].append(inf.count / cfg.n_inference)
                samples["in_sys"].append(in_system[0])
            yield env.timeout(1.0)

    env.process(arrivals())
    env.process(monitor())
    env.run(until=T_end)
    obs_days = cfg.sim_days - cfg.warmup_days
    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else float("nan")
    mean = lambda a: float(np.mean(a)) if len(a) else float("nan")
    return {
        "config": asdict(cfg), "service": service, "observed_days": obs_days,
        "arrived_per_day": rec["n_arrived"] / obs_days, "completed_per_day": len(rec["total"]) / obs_days,
        "camera_wait_min": {"mean": mean(rec["cam_wait"]), "p95": pct(rec["cam_wait"], 95)},
        "review_wait_min": {"mean": mean(rec["rev_wait"]), "p95": pct(rec["rev_wait"], 95)},
        "time_in_system_min": {"mean": mean(rec["total"]), "p95": pct(rec["total"], 95)},
        "mean_camera_queue": mean(samples["cam_q"]), "mean_review_queue": mean(samples["rev_q"]),
        "max_review_queue": int(max(samples["rev_q"])) if samples["rev_q"] else 0,
        "utilisation": {"cameras": mean(samples["cam_busy"]), "reviewers": mean(samples["rev_busy"]), "inference": mean(samples["inf_busy"])},
        "mean_in_system_L": mean(samples["in_sys"]),
        "littles_law": {"lambda_per_min": rec["n_arrived"] / (obs_days * cfg.day_min), "W_min": mean(rec["total"]),
                        "lambda_times_W": (rec["n_arrived"] / (obs_days * cfg.day_min)) * mean(rec["total"]), "L_observed": mean(samples["in_sys"])},
        "outcomes": {k: rec[k] for k in ("referred", "cleared", "missed", "no_review", "recaptures", "recapture_failed")},
    }


def sweep(base: SimConfig, cameras=(2, 3, 4, 5, 6), reviewers=(1, 2, 3, 4), reps: int = 3,
          target_review_wait_p95_min: float = 8 * 60, target_camera_wait_p95_min: float = 30) -> dict:
    """Grid over cameras x reviewers (mean of `reps` seeds). Cost proxy: cameras*1.0 + reviewers*1.0 'units' (assumption)."""
    rows = []
    for c in cameras:
        for r in reviewers:
            res = [run(SimConfig(**{**asdict(base), "n_cameras": c, "n_reviewers": r, "seed": base.seed + k})) for k in range(reps)]
            avg = lambda f: float(np.nanmean([f(x) for x in res]))
            rows.append({"cameras": c, "reviewers": r,
                         "camera_wait_p95_min": avg(lambda x: x["camera_wait_min"]["p95"]),
                         "review_wait_p95_min": avg(lambda x: x["review_wait_min"]["p95"]),
                         "review_wait_mean_min": avg(lambda x: x["review_wait_min"]["mean"]),
                         "time_in_system_mean_min": avg(lambda x: x["time_in_system_min"]["mean"]),
                         "util_cameras": avg(lambda x: x["utilisation"]["cameras"]), "util_reviewers": avg(lambda x: x["utilisation"]["reviewers"]),
                         "completed_per_day": avg(lambda x: x["completed_per_day"]), "arrived_per_day": avg(lambda x: x["arrived_per_day"]),
                         "meets_targets": bool(avg(lambda x: x["review_wait_min"]["p95"]) <= target_review_wait_p95_min
                                               and avg(lambda x: x["camera_wait_min"]["p95"]) <= target_camera_wait_p95_min)})
    ok = [r for r in rows if r["meets_targets"]]
    best = min(ok, key=lambda r: (r["cameras"] + r["reviewers"], r["review_wait_p95_min"])) if ok else None
    return {"rows": rows, "targets": {"review_wait_p95_min": target_review_wait_p95_min, "camera_wait_p95_min": target_camera_wait_p95_min},
            "cheapest_meeting_targets": best, "base_config": asdict(base)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/simulation.json")
    ap.add_argument("--metrics", default=None, help="optional metrics.json: use measured sensitivity/specificity as sim inputs")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    base = SimConfig()
    measured = {}
    if a.metrics and Path(a.metrics).is_file():
        m = json.loads(Path(a.metrics).read_text())
        sc = m["internal_holdout"]["screening"]
        base.sensitivity, base.specificity = sc["sensitivity"], sc["specificity"]
        measured["sensitivity"] = measured["specificity"] = "measured: classifier on the APTOS internal test split at the validation-selected threshold"
        q = m.get("quality", {}).get("pipeline_on_clean_test", {}).get("gate_plus_review_band")
        if q:
            base.p_quality_reject = q["n_recapture"] / q["n"]
            base.p_uncertain = q["n_human_review"] / q["n"]
            measured["p_quality_reject"] = "measured: share of APTOS test images rejected by the heuristic gate (an image-set rate, not an operational one)"
            measured["p_uncertain"] = "measured: share of APTOS test images in the confidence review band (an image-set rate)"
    out = {"baseline_run": run(base), "sweep": sweep(base, reps=a.reps), "inputs_taken_from_measurements": measured,
           "disclaimer": "Parameters not listed under inputs_taken_from_measurements are assumptions (see SimConfig). Nothing here is observed operational data."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["sweep"]["cheapest_meeting_targets"], indent=2))


if __name__ == "__main__":
    main()
