# One photograph, end to end

Everything in one connected story: a real photograph from the APTOS 2019 test split followed through the deployed system, stage by stage. All numbers below come from running the application's own `Predictor` (default ensemble) on this image when the guide was built (`report/guide/figures/walkthrough.json`). It is a real output, not a constructed example, and it was chosen because it is the interface's demonstration image, not because it looks good.

## The walkthrough

{{fig:walkthrough_images.png|The real example through the pipeline: the uploaded photograph, the preprocessed model view, the Grad-CAM map of the first ensemble member, and the lesion detections with the estimated disc, fovea and quadrant axes.|100|walk}}

### Stage 1: upload and validation

- **Input:** a JPEG/PNG file, {{val:walkthrough.width|d}} × {{val:walkthrough.height|d}} pixels; the dataset label for this eye is **grade {{val:walkthrough.dataset_grade|d}}** (proliferative DR). The system is not told this.
- **Operation:** decoding to an RGB array and checking it is a plausible photograph (`read_image`, `to_rgb_uint8`).
- **Concept:** an image is an $H \times W \times 3$ array (Part II).
- **Can go wrong:** wrong file type, a corrupt or tiny file. Each produces a clear message and no prediction.

### Stage 2: quality gate

- **Operation:** sharpness, brightness, contrast, saturation and field-of-view measurements compared with thresholds learned from training images.
- **Output here:** **{{val:walkthrough.quality_status}}**: "{{val:walkthrough.quality_reason}}".
- **Can go wrong:** the gate is heuristic. It would have passed a heavily compressed image (Part VI), which is why the robust models matter.
- **Evaluated by:** synthetic degradations only.

### Stage 3: preprocessing

- **Operation:** crop the black border, pad to a square, resize to 384 pixels (448 for B3), apply the circular mask, Ben Graham normalisation, ImageNet scaling.
- **Output:** a 3 × 384 × 384 tensor per model (panel 2 of {{ref:walk}} shows the plain stage).
- **Can go wrong:** a very dark image can defeat the border detection; resizing loses the finest detail.

### Stage 4: classification

- **Operation:** both ensemble members (robust EfficientNet-B0 and B3) process the image and its two mirror images; each member's averaged logits are temperature-scaled with its own validation-fitted temperature; the members' probabilities are averaged (`Predictor._ensemble_probs`).
- **Concepts:** convolution, transfer learning, softmax, temperature scaling, ensembles (Part III).
- **Output here** ({{ref:walkprobs}}): grade probabilities 0: {{val:walkthrough.probs.0|.3f}}, 1: {{val:walkthrough.probs.1|.3f}}, 2: {{val:walkthrough.probs.2|.3f}}, 3: {{val:walkthrough.probs.3|.3f}}, 4: {{val:walkthrough.probs.4|.3f}}. The most likely grade is **{{val:walkthrough.grade_name}}**, and the referable score is $p_2 + p_3 + p_4$ = **{{val:walkthrough.p_referable|.3f}}**.

{{fig:walkthrough_probs.png|The ensemble's grade probabilities for the example.|58|walkprobs}}

### Stage 5: decision

- **Operation:** compare the referable score with the ensemble's validation-chosen threshold ({{val:walkthrough.threshold|.3f}}) and review band (± {{val:walkthrough.review_margin|.3f}}).
- **Output here:** {{val:walkthrough.p_referable|.3f}} is above the threshold and outside the band, so the decision is **{{val:walkthrough.verdict}}**, with heuristic confidence *{{val:walkthrough.confidence}}*.
- **Evaluated by:** sensitivity, specificity and the other screening metrics on the test split (Parts V and VI).

### Stage 6: evidence and cross-check

- **Grad-CAM** (panel 3) highlights the regions that drove the first member's score.
- **Lesion detector** (panel 4): {{val:walkthrough.counts.MA|d}} microaneurysm, {{val:walkthrough.counts.HE|d}} haemorrhage, {{val:walkthrough.counts.EX|d}} hard-exudate and {{val:walkthrough.counts.SE|d}} cotton-wool regions; {{val:walkthrough.ex_near_macula|d}} exudate regions lie within one disc diameter of the estimated fovea, which triggers the macular warning.
- **Rule engine:** applying the calibrated criteria to those counts gives **rule grade {{val:walkthrough.rule_grade|d}}**, below the referable level.
- **Cross-check:** the classifier says referable and the rules do not, so the status is *{{val:walkthrough.cross_check}}*: the referral stands, and the report states that it is not corroborated by the lesion rules.

### Stage 7: what the user sees

The interface shows the photograph in its viewer, the recommendation and threshold chart, the evidence tabs and a downloadable PDF case report ({{ref:ui}}). The whole analysis took about {{val:walkthrough.latency_s|.1f}} seconds on a laptop CPU.

{{fig:ui_results_first_screen.png|The RetinaLab interface showing this analysis (screenshot of the running application).|100|ui}}

## What this example teaches

This single case shows the system working as designed, and its limits:

1. **The screening decision was right.** The dataset grade is 4, so the eye is referable, and the system recommended referral with a score well above the threshold.
2. **The severity grade was wrong.** The model's most likely grade was 2, not 4; only {{val:walkthrough.probs.4|.1%}} of the probability went to grade 4. This is the weakness seen in the per-grade results: the model rarely calls grade 4, partly because new vessels, the defining sign of proliferative disease, are thin structures that are hard to see at 384 pixels.
3. **The lesion rules under-called the case** (rule grade {{val:walkthrough.rule_grade|d}}), because the calibrated count thresholds are demanding and the detector cannot see new vessels at all. The cross-check correctly reported the disagreement instead of hiding it.
4. **The explanation is evidence, not proof.** The heat map and the detections help a clinician look at the right places; they do not establish that the model "understood" proliferative disease.

## One mental model of the whole system

{{fig:mental_model.png|The whole project on one page: biology defines the target, image processing standardises the input, the model produces calibrated probabilities, the evaluation fixes and measures the operating point, and the interface turns the result into a decision with evidence (original diagram).|100|mental}}

Read from left to right, {{ref:mental}} is the project. Diabetes damages retinal vessels and produces lesions; clinicians summarise them in five grades and refer grade 2 or worse. A fundus camera turns the retina into an array of pixels, which preprocessing makes uniform. An ImageNet-pretrained EfficientNet, fine-tuned on APTOS, turns the array into grade probabilities, and temperature scaling makes them better calibrated. A threshold chosen on validation data converts the referable score into a decision. That decision was measured on held-out test images with bootstrap intervals and checked on an external dataset, where it transferred less well. Around the decision, a quality gate, Grad-CAM, a lesion detector and a rule engine supply evidence and warnings, and the interface presents all of it as a screening aid, not a diagnosis.
