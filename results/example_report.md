Image: APTOS test image `042470a92154` (true grade 3), first grade-3 image in the test split.

### Image quality: **GOOD** (heuristic score 0.60)
sharpness, illumination, contrast and field of view are acceptable

### Screening result: **REFERABLE DR suspected**
- Decision: **refer**. P(referable) 0.96 >= threshold 0.63
- Classifier's predicted grade: **2 Moderate NPDR** (argmax of a CNN; it does not by itself explain why)
- P(referable) = **96.1%** (sum of P(grade 2-4); operating threshold 0.63)
- Grade probabilities: 0 No DR: 2.0%, 1 Mild NPDR: 1.9%, 2 Moderate NPDR: 76.9%, 3 Severe NPDR: 12.1%, 4 Proliferative DR: 7.1%
- Probabilities are temperature-scaled (T = 2.69, fitted on validation data); they are model scores, not clinically validated probabilities.
- Referral policy (illustrative): Refer to an eye specialist.

### Why: lesion evidence
**Lesion-rule grade: 2 (2 Moderate NPDR)**, rule-derived from detected lesions (data-calibrated (fitted on training split) thresholds)

- Lesions beyond microaneurysms were detected: 109 hard-exudate region(s). More than microaneurysms alone but the 4-2-1 haemorrhage criterion is not met: moderate NPDR.

| ICDR criterion | Status | Detail |
|---|---|---|
| Microaneurysms present | not met | 4 detected region(s) (needs >= 12) |
| Haemorrhages present | not met | 32 detected region(s) (needs >= 200) |
| Hard exudates present | met | 109 detected region(s) (needs >= 50) |
| Soft exudates (cotton-wool spots) present | not met | 3 detected region(s) (needs >= 5) |
| 4-2-1 rule, '4': haemorrhages in all 4 quadrants | not met | haemorrhage regions per quadrant: temporal-superior 7, temporal-inferior 1, nasal-superior 12, nasal-inferior 12 (needs >= 80 in each) |
| 4-2-1 rule, '2': venous beading in 2 quadrants | not assessed | no venous-beading detector in this project |
| 4-2-1 rule, '1': IRMA in 1 quadrant | not assessed | no IRMA detector in this project |
| Proliferative disease (neovascularisation / vitreous haemorrhage) | not assessed | no neovascularisation detector; grade 4 cannot be assigned by rules |

_The count thresholds above are calibrated to this detector's output on APTOS images; counts of detected regions are not clinical lesion counts (the detector fragments and over-detects lesions), so they should not be read as clinical cut-offs._

- Macular-involvement indicator: 31 hard-exudate region(s) within one disc diameter of the estimated fovea: possible macular involvement (macular oedema risk indicator, not a diagnosis; fovea position is model-estimated)

**Cross-check with the classifier:** The classifier and the lesion rules reach the same referable / non-referable decision.

_Localisation: Optic disc: segmented; fovea: predicted by the locator and consistent with the segmented disc._

_Lesion detections come from a model trained on IDRiD and applied to this image without lesion ground truth; they are evidence to be checked by a clinician, not verified findings. Venous beading, IRMA and neovascularisation are not assessed._

Overlay colours: red = microaneurysm, orange = haemorrhage, yellow = hard exudate, cyan = soft exudate; blue ring = optic disc, green cross = estimated fovea, white lines = quadrant axes.

**Grad-CAM** highlights image regions that influenced the classifier's score. It does not prove a lesion is present or that the model reasons like a clinician.

_This project is an academic research prototype for diabetic retinopathy screening. Its outputs are not a medical diagnosis and must not be used as a substitute for qualified clinical assessment._