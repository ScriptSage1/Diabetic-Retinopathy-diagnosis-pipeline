# Diabetic Retinopathy Screening: Roadmap

Every module follows the same loop: **learn a concept → build a piece of the real project with it → test that piece → move on.**
There is no separate "theory first" phase. Each deep learning idea is introduced exactly when the project needs it.

⭐ = needed for the demo + report deadline. Unmarked = stretch / come back later.

Biology and clinical topics are listed separately in [BIOLOGY.md](BIOLOGY.md).

---

## Phase 1: Getting the data ready

### Module 0: Kaggle setup, images as arrays, and the DR basics ⭐
- **Learn:** Kaggle notebooks, attaching datasets, switching on the GPU; NumPy arrays, shapes, indexing; an image is an `H × W × 3` array; RGB channels; reading images with OpenCV. Clinical side: eye anatomy (optic disc, macula, fovea, vessels), ICDR grades 0–4, lesion types (microaneurysms, haemorrhages, hard and soft exudates, IRMA, neovascularisation), the 4-2-1 rule, and what "referable DR" (grade ≥ 2) means.
- **Build:** a notebook that loads APTOS 2019, shows example images for each grade, splits an image into R/G/B channels, and plots the class distribution.
- **Test:** print shapes and dtypes of 10 random images; check the green channel really shows lesions most clearly; count images per grade and confirm the imbalance.
- **Done when:** you can look at a fundus image, roughly guess its grade, and point at the lesions.
- [ ] Done

### Module 1: Image preprocessing ⭐
- **Learn:** thresholding, binary masks, cropping, resizing and interpolation, Gaussian blur; why fundus images vary (camera, lighting, field of view); Ben Graham colour normalisation.
- **Build:** `preprocess.py`: crop the black border, apply a circular mask, resize, colour-normalise. Run it over APTOS and save the result as a Kaggle Dataset.
- **Test:** every output is exactly the target size; no black border remains; run it on 200 random images with zero crashes; a before/after grid looks consistent across dark and bright images.
- **Done when:** your preprocessed dataset is saved and reusable in any notebook.
- [ ] Done

### Module 2: Tensors and the data pipeline ⭐
- **Learn:** PyTorch tensors (shape, dtype, device), CPU vs GPU, the `N × C × H × W` layout, normalisation; `Dataset` and `DataLoader`; data augmentation (flips, rotations, brightness); stratified train/validation splits.
- **Build:** a `DRDataset` class, train/val transforms, and a stratified split of APTOS.
- **Test:** one batch has shape `(N, 3, H, W)` with sensible values; a grid of augmented images still looks like valid fundus photos; the grade distribution is the same in train and validation.
- **Done when:** you can explain what happens from "file on disk" to "batch on the GPU".
- [ ] Done

---

## Phase 2: The first model

### Module 3: Neural networks and the training loop ⭐
- **Learn:** neurons, weights, activations (ReLU, sigmoid, softmax); loss functions (cross-entropy); gradients and gradient descent; backpropagation and autograd; optimisers (SGD, Adam); the training loop (forward → loss → backward → step → zero_grad).
- **Build:** a simple fully connected network (MLP) on 64×64 images for **referable vs non-referable**, with your own training loop.
- **Test:** the classic sanity check is to **overfit a single batch** to ~100% accuracy (if it can't, there's a bug); then the training loss should go down on the full data. Expect poor validation results; that's the point of Module 4.
- **Done when:** you can write the training loop from memory and explain why the MLP struggles on images.
- [ ] Done

### Module 4: CNNs from scratch ⭐
- **Learn:** convolution (kernels, stride, padding, channels), feature maps, pooling, receptive field; batch norm, dropout; why CNNs beat MLPs on images; overfitting vs underfitting and how to read learning curves.
- **Build:** a small CNN (3–4 conv blocks) for referable DR at 224×224, replacing the MLP in the same training loop.
- **Test:** it beats the Module 3 MLP; plot train/val loss curves, deliberately make it overfit, then fix it with augmentation, dropout or weight decay; visualise first-layer filters and feature maps.
- **Done when:** you can compute a conv layer's output shape by hand and diagnose a learning curve.
- [ ] Done

### Module 5: Measuring like a clinician ⭐
- **Learn:** confusion matrix, **sensitivity**, **specificity**, precision, ROC curves and AUC, quadratic weighted kappa; why accuracy misleads with imbalanced classes; how the decision threshold trades sensitivity for specificity.
- **Build:** `metrics.py`: one function that produces the full evaluation report (confusion matrix, sensitivity/specificity, AUC, kappa, ROC plot) for any model.
- **Test:** check your numbers against scikit-learn on toy inputs; run it on the Module 4 CNN and record the baseline.
- **Done when:** you can explain why missing a referable case (false negative) is worse than a false alarm in screening.
- [ ] Done

### Module 6: Transfer learning ⭐
- **Learn:** ImageNet-pretrained models and why their features transfer; ResNet (skip connections) and EfficientNet; the `timm` library; freezing vs fine-tuning; learning-rate schedules; mixed precision (AMP); checkpointing.
- **Build:** fine-tune EfficientNet-B0 on APTOS, first for referable DR, then for all 5 grades.
- **Test:** it clearly beats your scratch CNN on the same split; reloading a saved checkpoint reproduces the same metrics.
- **Done when:** you can explain why pretraining helps when you only have a few thousand medical images.
- [ ] Done

### Module 7: A classifier strong enough for a clinic ⭐
- **Learn:** **patient-level splits** (data leakage between left and right eyes), higher resolution and progressive resizing, class imbalance (weighted loss/sampling), ordinal regression with optimised thresholds, test-time augmentation, small ensembles, choosing an operating point, external validation, bootstrap confidence intervals.
- **Build:** EfficientNet-B3/B4 trained on APTOS + EyePACS (resized), evaluated on IDRiD and Messidor-2, which it never trained on.
- **Test:** referable DR **sensitivity > 90% and specificity > 85% on external data**, with 95% confidence intervals.
- **Done when:** you hit the target and can explain why external validation matters more than validation accuracy.
- [ ] Done

---

## Phase 3: Making it trustworthy

### Module 8: Grad-CAM ⭐
- **Learn:** how Grad-CAM weights the last conv layer's feature maps by their gradients; PyTorch forward/backward hooks; Grad-CAM++.
- **Build:** Grad-CAM written **by hand with hooks**, then compared with `pytorch-grad-cam`; heatmap overlays on correct and incorrect predictions.
- **Test:** sanity check that randomising the model weights destroys the heatmap; on IDRiD images with lesion masks, measure how much of the heatmap falls on real lesions.
- **Done when:** you can spot when the model "cheats" by looking at borders or artefacts.
- [ ] Done

### Module 9: Calibrated confidence ⭐
- **Learn:** calibration, reliability diagrams, expected calibration error (ECE), temperature scaling; uncertainty and a "send to a human" threshold.
- **Build:** temperature scaling on the Module 7 model plus a referral rule for low-confidence cases.
- **Test:** ECE drops after scaling; the reliability diagram moves closer to the diagonal; low-confidence cases really do have higher error rates.
- **Done when:** you can explain why a 0.95 softmax output isn't automatically 95% confidence.
- [ ] Done

### Module 10: Image quality gate and enhancement ⭐
- **Learn:** focus measurement (Laplacian variance), illumination checks, field-of-view checks (circle detection); CLAHE on the LAB L-channel, illumination correction, non-local-means denoising.
- **Build:** `quality.py` returning *good / enhance / reject + reason*, e.g. "image is blurry, please recapture".
- **Test:** artificially blurred or darkened images get rejected with the right reason; compare with EyeQ quality labels; the Module 7 classifier scores higher on images that pass the gate.
- **Done when:** your pipeline refuses bad images instead of guessing on them.
- [ ] Done

---

## Phase 4: Segmentation, the "evidence" path

### Module 11: Vessel segmentation (your first U-Net)
- **Learn:** encoder-decoder networks, skip connections, upsampling; Dice and IoU; patch-based training on tiny datasets; the classical Frangi vessel filter.
- **Build:** a U-Net on DRIVE/HRF using `segmentation_models_pytorch`, plus a Frangi baseline.
- **Test:** Dice > 0.75 on the DRIVE test set; side-by-side comparison with Frangi.
- **Done when:** you understand how a U-Net turns an image into a pixel-wise map.
- [ ] Done

### Module 12: Lesions, optic disc and fovea ⭐
- **Learn:** multi-class segmentation; extreme pixel imbalance (Dice + focal loss); sliding-window inference on high-resolution images; heatmap regression to locate the fovea.
- **Build:** one U-Net on IDRiD (+ DDR) for hard exudates, soft exudates, haemorrhages and microaneurysms; an optic disc and fovea locator.
- **Test:** per-lesion Dice and precision-recall on the IDRiD test set; optic disc/fovea location error in pixels; visual overlays on 10 unseen images.
- **Done when:** you can explain why a model can get 99% pixel accuracy and still be useless.
- [ ] Done

### Module 13: Microaneurysms and neovascularisation (stretch)
- **Learn:** detection vs segmentation, centroid heatmaps, sub-pixel localisation, the FROC metric; signs of neovascularisation.
- **Build:** a full-resolution patch-based microaneurysm detector; NV detection on FGADR.
- **Test:** FROC score against the U-Net from Module 12.
- [ ] Done

---

## Phase 5: Reasoning and putting it together

### Module 14: Clinical rule engine ⭐
- **Learn:** feature engineering; dividing the retina into quadrants relative to the optic disc and fovea; turning the ICDR / 4-2-1 criteria into code; gradient boosting (LightGBM).
- **Build:** `rules.py`: lesion counts and areas per quadrant → ICDR grade **with human-readable reasons** (e.g. "haemorrhages in 4 quadrants → severe NPDR"); plus a LightGBM grader on the same features.
- **Test:** unit tests with hand-made inputs (only microaneurysms → grade 1; neovascularisation → grade 4); agreement (kappa) with true grades on IDRiD.
- **Done when:** the pipeline can explain its grade in clinical terms.
- [ ] Done

### Module 15: Fusion and the ablation study ⭐
- **Learn:** combining models (stacking), ablation study design, robustness testing.
- **Build:** fuse the CNN (Module 7) with the lesion path (Module 14); flag cases where they disagree for human review.
- **Test:** an ablation table on external data comparing:
  - CNN alone
  - lesion path alone
  - fused
  - each with and without the quality gate

  Repeat on synthetically degraded images (blur, low light, JPEG); compare with published results (Gulshan et al. 2016, IDx-DR).
- **Done when:** you can show, with numbers and confidence intervals, that the integrated pipeline beats each single technique.
- [ ] Done

### Module 16: Annotated report and demo app ⭐
- **Learn:** packaging inference into one function, Gradio basics, ONNX export.
- **Build:** `predict(image)` → a one-page report containing:
  - the original and enhanced image
  - lesion overlay and Grad-CAM
  - the grade with calibrated confidence
  - the ICDR criteria checklist
  - quality score and referral advice

  Wrap it in a Gradio app.
- **Test:** run end-to-end on 20 unseen images with no errors; measure time per image; time yourself reviewing a report (target: under 30 seconds).
- **Done when:** someone else can upload an image and understand the result without your help.
- [ ] Done

### Module 17: Screening workflow simulation ⭐
- **Learn:** discrete-event simulation with SimPy, queues and utilisation, Little's law, parameter sweeps.
- **Build:** a simulation of camera stations → upload over 2G/3G → inference server → ophthalmologist review queue → referral, for 100k+ patients a year (~400/day).
- **Test:** check one simple scenario against a hand calculation (Little's law); sweep the number of cameras and reviewer hours, and find the cheapest setup that keeps waiting times acceptable.
- **Done when:** you can recommend resources for a district program and back the recommendation with numbers.
- [ ] Done

### Module 18: The report ⭐
- **Learn:** how to present results honestly (confidence intervals, limitations).
- **Build:** the write-up:
  1. Introduction (DR burden in India, the screening gap)
  2. Related work
  3. Data and preprocessing
  4. Methods
  5. Results (metrics, external validation, ablation table, calibration plots, Grad-CAM examples)
  6. Workflow simulation
  7. Limitations and future work
  8. Conclusion
- **Test:** every number in the report traces back to a notebook output; every figure has a caption.
- [ ] Done

---

## Minimum path for the deadline
0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 (handcrafted checks only) → 12 → 14 → 15 → 16 → 17 → 18.
Skip 11 (use the Frangi filter instead) and 13 for now; come back to them after submission.

## Project layout
```
Diabetic Retinopathy CNN/
├── ROADMAP.md
├── notebooks/        # one notebook per module (M00_setup.ipynb, M01_preprocessing.ipynb, ...)
├── src/drscreen/     # code that grows module by module: preprocess, datasets, metrics, models, gradcam, quality, segmentation, rules, report
├── app/              # Gradio demo
├── sim/              # SimPy simulation
└── report/           # write-up, figures, tables
```
On Kaggle, upload `src/drscreen` as a Kaggle Dataset (or utility script) so every notebook can import it.

## Helpful resources
- PyTorch *Learn the Basics* tutorial
- 3Blue1Brown *Neural Networks* series
- Karpathy *Neural Networks: Zero to Hero* (micrograd lecture)
- fast.ai *Practical Deep Learning for Coders*
- CS231n course notes on CNNs
- Key papers:
  - Gulshan et al. 2016 (*JAMA*)
  - Grad-CAM (Selvaraju et al. 2017)
  - U-Net (Ronneberger et al. 2015)
  - EfficientNet (Tan & Le 2019)
  - *On Calibration of Modern Neural Networks* (Guo et al. 2017)
  - IDRiD (Porwal et al. 2018)

## Kaggle tips
- Run CPU-only work (EDA, preprocessing, LightGBM, SimPy) in **CPU sessions**; they don't use GPU quota.
- Preprocess once and save as a Kaggle Dataset.
- Debug on a small subset first; use **Save Version → Save & Run All** for long training runs.
- Use AMP and the T4 ×2 accelerator; checkpoint every epoch.
