"""Clinical reference rules and the lesion-evidence rule engine (ICDR severity scale, Wilkinson et al. 2003).

Honest scope:
  * The rule engine consumes lesion FEATURES produced by the IDRiD-trained lesion model (lesion.py). It only reasons about
    what that model can see: microaneurysms (MA), haemorrhages (HE), hard exudates (EX), soft exudates / cotton-wool spots (SE).
  * It CANNOT assess venous beading, IRMA or neovascularisation (no such detector exists here). Those criteria are reported as
    "not assessed", so the rule engine can never output grade 4 and can only reach grade 3 through the haemorrhage ('4') part
    of the 4-2-1 rule.
  * Detection counts are not clinical counts (merged / split / missed lesions, domain shift to APTOS), so the literal ICDR
    count thresholds are an approximation. A data-calibrated variant (thresholds fitted on the TRAINING split) is provided and
    evaluated separately.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .metrics import GRADE_NAMES, REFERABLE_MIN_GRADE

GRADE_DESCRIPTIONS = {
    0: "No apparent retinopathy: no visible abnormalities on the ICDR scale.",
    1: "Mild non-proliferative DR: microaneurysms only.",
    2: "Moderate non-proliferative DR: more than microaneurysms alone, but less than severe NPDR.",
    3: "Severe non-proliferative DR: meets any 4-2-1 criterion (haemorrhages in 4 quadrants, venous beading in 2, or IRMA in 1) without new vessels.",
    4: "Proliferative DR: neovascularisation and/or vitreous/preretinal haemorrhage.",
}

# Illustrative policy from the project documentation (docs/BIOLOGY.md, section 8): not clinical guidance.
REFERRAL_POLICY = {
    0: ("Routine", "Re-screen in about 12 months."),
    1: ("Routine", "Re-screen in about 12 months."),
    2: ("Refer", "Refer to an eye specialist."),
    3: ("Urgent", "Refer urgently to an eye specialist."),
    4: ("Urgent", "Refer urgently to an eye specialist."),
}

ICDR_CRITERIA_REFERENCE = [
    ("Grade 0", "No abnormalities."),
    ("Grade 1", "Microaneurysms only."),
    ("Grade 2", "More than microaneurysms only, less than severe NPDR."),
    ("Grade 3", "Any of: >=20 intraretinal haemorrhages in each of 4 quadrants; definite venous beading in >=2 quadrants; "
                "prominent IRMA in >=1 quadrant; and no signs of proliferative disease."),
    ("Grade 4", "Neovascularisation and/or vitreous or preretinal haemorrhage."),
]


def is_referable(grade: int) -> bool:
    return int(grade) >= REFERABLE_MIN_GRADE


def grade_name(grade: int) -> str:
    return GRADE_NAMES[int(grade)]


def referral_advice(grade: int) -> dict:
    level, text = REFERRAL_POLICY[int(grade)]
    return {"level": level, "advice": text, "source": "illustrative policy from the project's docs/BIOLOGY.md; not clinical guidance"}


# ----------------------------------------------------------------------------- rule engine
@dataclass
class RuleParams:
    """Minimum detected-region counts to count a lesion type as present, and the per-quadrant haemorrhage count of the '4' rule."""
    min_ma: int = 1
    min_he: int = 1
    min_ex: int = 1
    min_se: int = 1
    he_per_quadrant: int = 20   # ICDR '4' of the 4-2-1 rule
    name: str = "ICDR-literal"

    def to_dict(self):
        return asdict(self)


@dataclass
class RuleResult:
    grade: int
    referable: bool
    reasons: list
    criteria: list
    dme_flag: bool
    dme_detail: str
    params: dict
    not_assessed: list

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [f"**Lesion-rule grade: {self.grade} ({GRADE_NAMES[self.grade]})**, rule-derived from detected lesions ({self.params['name']} thresholds)", ""]
        lines += [f"- {r}" for r in self.reasons]
        lines += ["", "| ICDR criterion | Status | Detail |", "|---|---|---|"]
        for c in self.criteria:
            lines.append(f"| {c['name']} | {c['status'].replace('_', ' ')} | {c['detail']} |")
        if str(self.params.get("name", "")).startswith("data-calibrated"):
            lines += ["", "_The count thresholds above are calibrated to this detector's output on APTOS images; counts of detected regions are not clinical lesion counts "
                          "(the detector fragments and over-detects lesions), so they should not be read as clinical cut-offs._"]
        lines += ["", f"- Macular-involvement indicator: {self.dme_detail}"]
        return "\n".join(lines)


def _q(features: dict, k: str) -> list[int]:
    return [int(features.get(f"{k}_q{q}", 0)) for q in range(4)]


def apply_rules(features: dict, params: RuleParams | None = None) -> RuleResult:
    """Map lesion features to an ICDR grade (0-3) with the reasons. Missing/failed localisation => features are unreliable."""
    p = params or RuleParams()
    ma, he, ex, se = (int(features.get(f"{k}_count", 0)) for k in ("MA", "HE", "EX", "SE"))
    he_q = _q(features, "HE")
    he_min = int(features.get("HE_min_quadrant_count", min(he_q)))
    four_met = he_min >= p.he_per_quadrant
    quadrant_names = ["temporal-superior", "temporal-inferior", "nasal-superior", "nasal-inferior"]
    crit = [
        {"name": "Microaneurysms present", "status": "met" if ma >= p.min_ma else "not_met", "detail": f"{ma} detected region(s) (needs >= {p.min_ma})"},
        {"name": "Haemorrhages present", "status": "met" if he >= p.min_he else "not_met", "detail": f"{he} detected region(s) (needs >= {p.min_he})"},
        {"name": "Hard exudates present", "status": "met" if ex >= p.min_ex else "not_met", "detail": f"{ex} detected region(s) (needs >= {p.min_ex})"},
        {"name": "Soft exudates (cotton-wool spots) present", "status": "met" if se >= p.min_se else "not_met", "detail": f"{se} detected region(s) (needs >= {p.min_se})"},
        {"name": "4-2-1 rule, '4': haemorrhages in all 4 quadrants", "status": "met" if four_met else "not_met",
         "detail": "haemorrhage regions per quadrant: " + ", ".join(f"{n} {c}" for n, c in zip(quadrant_names, he_q)) + f" (needs >= {p.he_per_quadrant} in each)"},
        {"name": "4-2-1 rule, '2': venous beading in 2 quadrants", "status": "not_assessed", "detail": "no venous-beading detector in this project"},
        {"name": "4-2-1 rule, '1': IRMA in 1 quadrant", "status": "not_assessed", "detail": "no IRMA detector in this project"},
        {"name": "Proliferative disease (neovascularisation / vitreous haemorrhage)", "status": "not_assessed", "detail": "no neovascularisation detector; grade 4 cannot be assigned by rules"},
    ]
    reasons: list[str] = []
    if four_met:
        grade = 3
        reasons.append(f"Haemorrhage-like regions found in all four quadrants (at least {p.he_per_quadrant} each): meets the '4' part of the 4-2-1 rule, so severe NPDR.")
    elif he >= p.min_he or ex >= p.min_ex or se >= p.min_se:
        grade = 2
        found = []
        if he >= p.min_he:
            found.append(f"{he} haemorrhage region(s) (in {int(features.get('HE_quadrants_ge1', 0))} of 4 quadrants)")
        if ex >= p.min_ex:
            found.append(f"{ex} hard-exudate region(s)")
        if se >= p.min_se:
            found.append(f"{se} soft-exudate region(s)")
        reasons.append("Lesions beyond microaneurysms were detected: " + "; ".join(found) + ". More than microaneurysms alone but the 4-2-1 haemorrhage criterion is not met: moderate NPDR.")
    elif ma >= p.min_ma:
        grade = 1
        reasons.append(f"Only microaneurysms were detected ({ma} region(s)); no haemorrhages, exudates or cotton-wool spots: mild NPDR.")
    else:
        grade = 0
        reasons.append("No microaneurysms, haemorrhages or exudates were detected by the lesion model: no apparent retinopathy (it may miss subtle lesions).")
    near = int(features.get("EX_within_1DD", 0))
    dme = near >= 1
    dme_detail = (f"{near} hard-exudate region(s) within one disc diameter of the estimated fovea: possible macular involvement (macular oedema risk indicator, "
                  "not a diagnosis; fovea position is model-estimated)" if dme else "no hard exudates detected within one disc diameter of the estimated fovea")
    if not features.get("loc_ok", 1):
        reasons.insert(0, "WARNING: the image could not be analysed by the lesion model; the result below is not meaningful.")
    return RuleResult(grade, is_referable(grade), reasons, crit, dme, dme_detail, p.to_dict(), [c["name"] for c in crit if c["status"] == "not_assessed"])


def _arrays(features) -> dict:
    """Column arrays used by the vectorised rules (accepts a DataFrame or a list of feature dicts)."""
    if hasattr(features, "columns"):
        get = lambda c: features[c].to_numpy(float) if c in features.columns else np.zeros(len(features))
    else:
        rows = list(features)
        get = lambda c: np.array([r.get(c, 0) for r in rows], float)
    return {"MA": get("MA_count"), "HE": get("HE_count"), "EX": get("EX_count"), "SE": get("SE_count"), "hmin": get("HE_min_quadrant_count")}


def _grade_from_arrays(a: dict, p: RuleParams) -> np.ndarray:
    return np.where(a["hmin"] >= p.he_per_quadrant, 3,
                    np.where((a["HE"] >= p.min_he) | (a["EX"] >= p.min_ex) | (a["SE"] >= p.min_se), 2,
                             np.where(a["MA"] >= p.min_ma, 1, 0))).astype(int)


def grade_vector(features, params: RuleParams) -> np.ndarray:
    """Vectorised apply_rules()->grade over a DataFrame / list of feature dicts (same logic, no reasons)."""
    return _grade_from_arrays(_arrays(features), params)


DEFAULT_FIT_GRID = {"min_ma": [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200], "min_he": [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200],
                    "min_ex": [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 400], "min_se": [1, 2, 3, 5, 8, 12, 20],
                    "he_per_quadrant": [3, 5, 8, 12, 20, 30, 50, 80]}


def fit_rule_params(features, grades, grid: dict | None = None, passes: int = 4) -> RuleParams:
    """Fit the count thresholds by coordinate descent (maximise QWK against the labels) on the TRAINING split only.
    Starts from the literal ICDR counts; a wide grid is needed because detected-region counts are far larger than clinical counts."""
    from .metrics import quadratic_weighted_kappa

    grid = grid or DEFAULT_FIT_GRID
    y = np.asarray(grades).astype(int)
    arr = _arrays(features)
    cur = RuleParams().to_dict()
    cur["name"] = "data-calibrated (fitted on training split)"
    score = lambda d: quadratic_weighted_kappa(y, _grade_from_arrays(arr, RuleParams(**d)))
    best = score(cur)
    best = -2.0 if np.isnan(best) else best
    for _ in range(passes):
        improved = False
        for k, vals in grid.items():
            for v in vals:
                trial = {**cur, k: v}
                sc = score(trial)
                if not np.isnan(sc) and sc > best + 1e-9:
                    best, cur, improved = sc, trial, True
        if not improved:
            break
    return RuleParams(**cur)


def lesion_evidence(analysis: dict | None = None, params: RuleParams | None = None) -> dict:
    """Evidence block for the app/report. Without a lesion analysis it states plainly that none was computed."""
    if analysis is None:
        return {"available": False,
                "reason": "No lesion model was available for this run (lesion_unet.pt / locator.pt not found), so lesion counts, quadrant analysis and "
                          "4-2-1 criteria were NOT computed, and the grade shown is the classifier's, not a rule-derived one."}
    res = apply_rules(analysis["features"], params)
    loc = analysis.get("location") or {}
    if loc.get("od_source") == "segmentation" and loc.get("fovea_source") == "regressor":
        loc_note = "Optic disc: segmented; fovea: predicted by the locator and consistent with the segmented disc."
    elif loc.get("fovea_source") == "anatomical prior":
        loc_note = ("Optic disc: segmented; the locator's fovea estimate disagreed with the segmented disc, so the fovea was placed by an anatomical prior "
                    "(about 2.5 disc diameters from the disc toward the centre of the retina). Quadrant and macular statements are therefore approximate.")
    elif loc:
        loc_note = "No optic disc was segmented; positions come from the locator alone and may be unreliable. Quadrant and macular statements are approximate."
    else:
        loc_note = ""
    return {"available": True, "rule": res.to_dict(), "markdown": res.to_markdown(), "features": analysis["features"], "location_note": loc_note,
            "caveat": "Lesion detections come from a model trained on IDRiD and applied to this image without lesion ground truth; they are "
                      "evidence to be checked by a clinician, not verified findings. Venous beading, IRMA and neovascularisation are not assessed."}
