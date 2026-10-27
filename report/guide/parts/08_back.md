# Reference and viva preparation

Viva questions with model answers, a glossary of every important term, and the references cited in the text.

## Viva questions and answers

The questions run from foundations to the hardest questions about trust and limitations. The answers are short enough to say aloud and are limited to what the project actually did.

### The biological problem

::: viva Q1. What is diabetic retinopathy, and why screen for it?
It is damage to the small blood vessels of the retina caused by years of high blood glucose. Vessels weaken, leak and close, producing microaneurysms, haemorrhages and exudates, and in the worst stage fragile new vessels. Early stages cause no symptoms, but treatment works best before vision is lost, so people with diabetes are photographed regularly and those with significant disease are referred.
:::

::: viva Q2. What do the ICDR grades mean, and why is "referable" grade 2 or higher?
Grade 0 is no retinopathy, 1 is mild (microaneurysms only), 2 moderate, 3 severe (the 4-2-1 rule) and 4 proliferative (new vessels or vitreous haemorrhage). Moderate disease or worse warrants specialist assessment, so the project defines referable as grade ≥ 2, the definition in its roadmap and a common one in the screening literature. It is coded once as `REFERABLE_MIN_GRADE = 2`.
:::

::: viva Q3. What is the difference between screening and diagnosis?
Screening sorts many people without symptoms into "refer" and "re-screen later"; diagnosis is a clinician's judgement about an individual, using an examination and other information. Our system produces model predictions from one photograph, which is a screening aid at most and never a diagnosis.
:::

::: viva Q4. Which lesions can your system detect, and which not?
The lesion network segments microaneurysms, haemorrhages, hard exudates and cotton-wool spots, plus the optic disc. It cannot detect venous beading, IRMA or neovascularisation, so the rule engine marks those criteria "not assessed" and can never assign grade 4 by itself.
:::

### Images, data and preprocessing

::: viva Q5. Why apply Ben Graham normalisation?
Fundus images vary in colour and illumination between cameras. Subtracting a heavily blurred copy removes the slow background variation and keeps the fine structures (vessels, lesions), so the network learns disease patterns rather than camera differences. The cost is an unnatural look and amplified noise.
:::

::: viva Q6. Why is augmentation used only for training?
Augmentation creates varied training examples so the model learns invariant features. Evaluation must be repeatable and must reflect real inputs, so validation, test and live images use only the fixed preprocessing.
:::

::: viva Q7. How did you prevent data leakage without patient IDs?
Images with identical file hashes or near-identical perceptual hashes were grouped and each group kept in one split, using a stratified, group-aware split; an assertion fails if a group crosses splits. This catches duplicate photographs but not different photographs of the same patient, so the internal test results may be somewhat optimistic. The EyePACS experiment, which has patient IDs, was split by patient.
:::

::: viva Q8. The data are imbalanced. What did you do about it?
Training used class-weighted cross-entropy with weights proportional to the inverse square root of class frequency, checkpoints were selected on quadratic weighted kappa rather than accuracy, and per-grade metrics are reported. The rare severe grades remain the weakest.
:::

### Model and training

::: viva Q9. Why EfficientNet-B0 and why transfer learning?
B0 offers strong accuracy for few parameters (about 4 million), fits a free T4 GPU and runs on a CPU. Pre-trained ImageNet features (edges, textures) transfer to retinal images, so far fewer labelled images are needed than when training from scratch. Our scratch CNN reached a clearly lower QWK.
:::

::: viva Q10. Walk me through one training step.
A batch of 24 augmented, normalised images goes through the network (forward pass) to give logits; class-weighted cross-entropy compares them with the true grades; back-propagation computes the gradient of the loss for every weight; the gradient norm is clipped to 5; AdamW updates the weights; the learning-rate schedule advances. After each epoch the validation QWK decides whether to save `best.pt`.
:::

::: viva Q11. How did you avoid over-fitting?
Augmentation, dropout of 0.3 before the classifier, weight decay of 0.01, and early stopping with patience 8 on validation QWK, keeping the best epoch ({{val:run.best_epoch|d}} of {{val:run.epochs_completed|d}}).
:::

::: viva Q12. What is the difference between the score, the probability and the confidence in your interface?
The predicted grade is the argmax of the softmax. The referable score is the sum of the probabilities of grades 2–4 after temperature scaling. It is calibrated on APTOS validation data, but it is not a verified probability for other populations. The confidence label is a heuristic based on the distance of the score from the threshold, adjusted for image quality and lesion disagreement. It is not the probability of being right.
:::

### Evaluation

::: viva Q13. Why not report accuracy alone?
With imbalanced classes a useless model can have high accuracy, and accuracy hides which mistakes are made. In screening a missed referable eye matters more than a false alarm, so sensitivity and specificity, with their counts and threshold, are the primary measures, together with threshold-free ROC-AUC and PR-AUC.
:::

::: viva Q14. How was the threshold chosen, and why not 0.5?
On the validation set, as the highest threshold that keeps sensitivity at or above 0.90, which maximises specificity under that floor. It was then frozen for the test set. 0.5 has no special meaning for a screening trade-off. The choice prioritises not missing referable eyes.
:::

::: viva Q15. What does QWK measure, and why can it mislead?
It is chance-corrected agreement between true and predicted grades, penalising errors by the square of their distance. Because most errors are off by one and grade 0 dominates, QWK was {{val:internal_holdout.grades.qwk|.2f}} even though grade 3 was recognised in under half of cases.
:::

::: viva Q16. What does your confidence interval mean?
It comes from the percentile bootstrap: the test images were resampled with replacement 1,000 times and the 2.5th and 97.5th percentiles of the metric taken. It describes sampling uncertainty for images like these, assuming the images are independent. It says nothing about other populations.
:::

::: viva Q17. What is calibration and how did you check it?
A calibrated model's 80% scores are right about 80% of the time. Temperature scaling, fitted on validation data, reduced the five-grade expected calibration error on the test set from {{val:internal_holdout.calibration_uncalibrated.ece_top_label_5grade|.3f}} to {{val:internal_holdout.calibration.ece_top_label_5grade|.3f}}; reliability diagrams show the same.
:::

### Explainability and components

::: viva Q18. What does Grad-CAM show, and what does it not?
It weights the last convolutional feature maps by the averaged gradients of the class score and shows where the evidence for the predicted grade came from, on a 12 × 12 grid. It is not lesion detection, it is coarse, and a plausible map does not prove correct reasoning. A randomisation check confirmed only that the maps depend on the learned weights.
:::

::: viva Q19. Did the lesion pathway improve accuracy?
No. Fusing lesion features with the classifier did not significantly change ROC-AUC on the test split. Its value is explanation: it provides countable findings, an ICDR checklist and a cross-check that flags disagreements for human review.
:::

::: viva Q20. How were the robust model and the ensemble chosen?
By rules written down before the results were seen and applied by scripts. The robust model had to be non-inferior on clean images and improve balanced accuracy on degraded ones by at least 0.05. The ensemble was selected on validation score and had to be non-inferior on the test split. Both passed; the EyePACS model and lesion v2 did not pass their rules and were not adopted.
:::

### Results, limitations and trust

::: viva Q21. Your internal sensitivity is 0.93. Should a clinic trust the system?
No. That figure comes from images of the same source as the training data. On IDRiD, a different camera and population, sensitivity fell to {{val:external_idrid.rows.0.sensitivity|.2f}}. There is no patient-level split, no prospective study and no regulatory review. The system is a research prototype; a clinic would need local external validation, threshold re-calibration and clinical oversight.
:::

::: viva Q22. Why did performance drop on IDRiD?
The likely reasons are domain shift (camera, image processing, population, grading protocol) and a threshold tuned to APTOS. The ROC-AUC stayed fairly high, so ranking partly survived while the operating point moved. With about 100 images we cannot pin down which cause dominates; adding EyePACS data raised the IDRiD ROC-AUC, which supports, but does not prove, the role of training-data diversity.
:::

::: viva Q23. How do your results compare with published systems?
Gulshan et al. reported sensitivities around 87–90% at specificities around 98% on EyePACS-1 and Messidor-2, with far larger training sets and a definition that also included macular oedema [35]. The IDx-DR pivotal trial evaluated an autonomous system prospectively in primary care [36]. Our internal numbers look similar, but they are not comparable: different data, definitions and, crucially, no prospective or multi-site validation.
:::

::: viva Q24. If you had three more months, what would you do first?
Validate externally on larger independent datasets with local threshold calibration, add more diverse training data with patient-level splits, and replace the heuristic quality gate with a model trained on real quality labels. Then add detectors for new vessels and venous beading so the rule engine can assess all ICDR criteria.
:::

## Glossary

| Term | Meaning |
|---|---|
| AdamW | an optimiser that adapts each weight's step size from its recent gradients and applies decoupled weight decay |
| augmentation | random, label-preserving changes to training images (rotation, flips, brightness) |
| AUPR / PR-AUC | area under the precision-recall curve (average precision) |
| back-propagation | the chain-rule algorithm that computes all gradients of the loss efficiently |
| balanced accuracy | the mean of sensitivity and specificity |
| batch / epoch | a group of images processed together / one pass over all training images |
| Ben Graham normalisation | subtracting a blurred copy of the image to remove illumination and colour differences |
| bootstrap | estimating uncertainty by recomputing a metric on resamples drawn with replacement |
| Brier score | mean squared difference between a predicted probability and the 0/1 outcome |
| calibration | agreement between predicted probabilities and observed frequencies |
| channel | one colour component (R, G or B) of an image, or one feature map in a CNN |
| CLAHE | contrast-limited adaptive histogram equalisation, a local contrast enhancement |
| CNN | convolutional neural network: a network built from sliding filters |
| confusion matrix | table of true versus predicted classes (TP, FP, TN, FN in the binary case) |
| cotton-wool spot | fluffy pale patch caused by local ischaemia of nerve fibres (soft exudate) |
| cross-entropy | loss equal to minus the log of the probability given to the correct class |
| Dice / IoU | overlap measures between a predicted and a reference mask |
| dropout | randomly zeroing values during training to prevent over-reliance on single features |
| early stopping | ending training when the validation metric stops improving |
| ECE | expected calibration error: the weighted average gap between confidence and accuracy |
| ensemble | combining the predictions of several models |
| external validation | evaluation on data from a different source than the training data |
| feature map | the output of one convolutional filter over an image |
| fovea / macula | the centre of sharp vision / the region of the retina around it |
| fundus photograph | a photograph of the back of the eye taken through the pupil |
| gradient | the slope of the loss with respect to a weight; the direction of steepest increase |
| Grad-CAM | heat map of the regions that most increased a class score, from gradients and feature maps |
| haemorrhage | blood leaked into the retina |
| hard exudate | yellow lipid and protein deposit left by leaking vessels |
| ICDR scale | the five-level International Clinical Diabetic Retinopathy severity scale |
| inference | using a trained model on new inputs |
| IRMA | intraretinal microvascular abnormalities, abnormal small vessels within the retina |
| leakage | information from evaluation data influencing training or selection |
| logit | a raw, unbounded output score of the network before softmax |
| microaneurysm | small bulge in a capillary wall; the earliest visible sign of DR |
| neovascularisation | growth of fragile new vessels; defines proliferative DR |
| NPV / PPV | the share of negative / positive results that are correct |
| optic disc | where the optic nerve and the retinal vessels leave the eye; the brightest round structure in a fundus photograph |
| over-fitting | memorising training data so that performance on new data suffers |
| prevalence | the share of a population that has the condition |
| QWK | quadratic weighted kappa: chance-corrected agreement on an ordinal scale |
| referable DR | in this project, ICDR grade 2 or higher |
| ROC curve / ROC-AUC | sensitivity against false-positive rate over all thresholds / the area under it |
| segmentation | assigning a class to every pixel |
| sensitivity / specificity | share of positives found / share of negatives correctly cleared |
| softmax | function turning logits into probabilities that sum to 1 |
| temperature scaling | dividing logits by one fitted number to calibrate probabilities |
| test-time augmentation | averaging predictions over transformed copies of an input |
| threshold | the score above which a case is called positive |
| transfer learning | starting from a network pre-trained on another task and fine-tuning it |
| U-Net | encoder-decoder segmentation network with skip connections |
| validation set | data used for choices (checkpoint, temperature, threshold), not for training or final measurement |

## References

1. Wilkinson CP, Ferris FL, Klein RE, et al. Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales. *Ophthalmology*. 2003;110(9):1677–1682.
2. Cheung N, Mitchell P, Wong TY. Diabetic retinopathy. *The Lancet*. 2010;376(9735):124–136.
3. Antonetti DA, Klein R, Gardner TW. Diabetic retinopathy. *New England Journal of Medicine*. 2012;366(13):1227–1239.
4. Teo ZL, Tham YC, Yu M, et al. Global prevalence of diabetic retinopathy and projection of burden through 2045: systematic review and meta-analysis. *Ophthalmology*. 2021;128(11):1580–1591.
5. Purves D, Augustine GJ, Fitzpatrick D, et al., eds. *Neuroscience*. 6th ed. Oxford University Press / Sinauer; 2018.
6. Abràmoff MD, Garvin MK, Sonka M. Retinal imaging and image analysis. *IEEE Reviews in Biomedical Engineering*. 2010;3:169–208.
7. Asia Pacific Tele-Ophthalmology Society. APTOS 2019 Blindness Detection (Kaggle competition). https://www.kaggle.com/c/aptos2019-blindness-detection
8. Porwal P, Pachade S, Kamble R, et al. Indian Diabetic Retinopathy Image Dataset (IDRiD): a database for diabetic retinopathy screening research. *Data*. 2018;3(3):25.
9. Decencière E, Cazuguel G, Zhang X, et al. TeleOphta: machine learning and image processing methods for teleophthalmology. *IRBM*. 2013;34(2):196–203.
10. Kaggle and EyePACS. Diabetic Retinopathy Detection (Kaggle competition, 2015). https://www.kaggle.com/c/diabetic-retinopathy-detection
11. Graham B. Kaggle Diabetic Retinopathy Detection competition report. University of Warwick; 2015.
12. Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
13. LeCun Y, Bottou L, Bengio Y, Haffner P. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*. 1998;86(11):2278–2324.
14. Rumelhart DE, Hinton GE, Williams RJ. Learning representations by back-propagating errors. *Nature*. 1986;323:533–536.
15. Deng J, Dong W, Socher R, Li LJ, Li K, Fei-Fei L. ImageNet: a large-scale hierarchical image database. *IEEE Conference on Computer Vision and Pattern Recognition*. 2009:248–255.
16. Tan M, Le QV. EfficientNet: rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (PMLR 97)*. 2019:6105–6114.
17. Sandler M, Howard A, Zhu M, Zhmoginov A, Chen LC. MobileNetV2: inverted residuals and linear bottlenecks. *IEEE Conference on Computer Vision and Pattern Recognition*. 2018:4510–4520.
18. Hu J, Shen L, Sun G. Squeeze-and-excitation networks. *IEEE Conference on Computer Vision and Pattern Recognition*. 2018:7132–7141.
19. Kingma DP, Ba J. Adam: a method for stochastic optimization. *International Conference on Learning Representations*. 2015.
20. Loshchilov I, Hutter F. Decoupled weight decay regularization. *International Conference on Learning Representations*. 2019.
21. Loshchilov I, Hutter F. SGDR: stochastic gradient descent with warm restarts. *International Conference on Learning Representations*. 2017.
22. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning (PMLR 70)*. 2017:1321–1330.
23. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: visual explanations from deep networks via gradient-based localization. *IEEE International Conference on Computer Vision*. 2017:618–626.
24. Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity checks for saliency maps. *Advances in Neural Information Processing Systems 31*. 2018.
25. Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, LNCS 9351. 2015:234–241.
26. Dice LR. Measures of the amount of ecologic association between species. *Ecology*. 1945;26(3):297–302.
27. Fawcett T. An introduction to ROC analysis. *Pattern Recognition Letters*. 2006;27(8):861–874.
28. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*. 2015;10(3):e0118432.
29. Altman DG, Bland JM. Diagnostic tests 1: sensitivity and specificity. *BMJ*. 1994;308:1552.
30. Altman DG, Bland JM. Diagnostic tests 2: predictive values. *BMJ*. 1994;309:102.
31. Cohen J. Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit. *Psychological Bulletin*. 1968;70(4):213–220.
32. Efron B, Tibshirani RJ. *An Introduction to the Bootstrap*. Chapman & Hall; 1993.
33. Brier GW. Verification of forecasts expressed in terms of probability. *Monthly Weather Review*. 1950;78(1):1–3.
34. Naeini MP, Cooper GF, Hauskrecht M. Obtaining well calibrated probabilities using Bayesian binning. *Proceedings of the AAAI Conference on Artificial Intelligence*. 2015:2901–2907.
35. Gulshan V, Peng L, Coram M, et al. Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. *JAMA*. 2016;316(22):2402–2410.
36. Abràmoff MD, Lavin PT, Birch M, Shah N, Folk JC. Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices. *npj Digital Medicine*. 2018;1:39.
37. Paszke A, Gross S, Massa F, et al. PyTorch: an imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems 32*. 2019.
38. Wightman R. PyTorch Image Models (timm). 2019. doi:10.5281/zenodo.4414861
39. Abid A, Abdalla A, Abid A, Khan D, Alfozan A, Zou J. Gradio: hassle-free sharing and testing of ML models in the wild. arXiv:1906.02569. 2019.
40. SimPy: discrete event simulation for Python. https://simpy.readthedocs.io
41. Little JDC. A proof for the queuing formula: L = λW. *Operations Research*. 1961;9(3):383–387.
42. Zuiderveld K. Contrast limited adaptive histogram equalization. In: *Graphics Gems IV*. Academic Press; 1994:474–485.
43. Micikevicius P, Narang S, Alben J, et al. Mixed precision training. *International Conference on Learning Representations*. 2018.
44. Srivastava N, Hinton G, Krizhevsky A, Sutskever I, Salakhutdinov R. Dropout: a simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*. 2014;15:1929–1958.
45. Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning (PMLR 37)*. 2015:448–456.
46. Ramachandran P, Zoph B, Le QV. Searching for activation functions. arXiv:1710.05941. 2017.
47. Yosinski J, Clune J, Bengio Y, Lipson H. How transferable are features in deep neural networks? *Advances in Neural Information Processing Systems 27*. 2014.

Datasets and online resources were accessed during the project in October 2026. Results of other studies quoted in this guide are theirs, not ours, and were obtained under different conditions.
