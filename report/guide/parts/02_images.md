# From a photograph to numbers

How a retinal photograph becomes the array of numbers a neural network can learn from: pixels and channels, the preprocessing and augmentation the project actually applies, and how the dataset is turned into training, validation and test sets without leaking information.

## Pixels, channels and arrays

### An image is a grid of numbers

::: define Pixel, channel, array
A digital image is a rectangular grid of **pixels** (picture elements). A colour pixel stores three numbers, the intensities of **red, green and blue** light (the three **channels**), usually as whole numbers from 0 (none) to 255 (full). A colour image of height $H$ and width $W$ is therefore an **array** of shape $H \times W \times 3$, a three-dimensional table of numbers.
:::

{{ref:pixels}} zooms into a real fundus image until individual pixels are visible, then prints the actual values of nine of them. The pixels inside a vessel have lower values, especially in the green channel, than the surrounding retina; dark lesions such as haemorrhages behave the same way, and bright lesions such as exudates have higher values. Everything a model "sees" is patterns in such numbers.

{{fig:pixels.png|From photograph to numbers, using a real image from the project. Left: the 384 × 384 preprocessed image the model works on (yellow box: the region enlarged). Middle: 12 × 12 pixels. Right: the actual red, green and blue values of the top-left 3 × 3 pixels.|100|pixels}}

::: example A 3 × 3 image as the computer receives it
A grey-scale image three pixels wide and three high is just a 3 × 3 table, for instance `[[12, 40, 41], [13, 200, 45], [11, 38, 40]]`: a bright spot in the middle of a dark patch. A colour version adds a third axis holding (R, G, B) for each pixel, so its shape is 3 × 3 × 3. A real fundus photograph of 3,216 × 2,136 pixels holds about 20.6 million numbers.
:::

### How the array's shape changes on the way to the model

Fundus photographs arrive in many sizes, but a neural network needs a fixed input size. On its way through the project the array changes shape like this (the numbers are those of the image in {{ref:prep}}):

{{tab:shapes|Shape of the data at each stage, for one real image.}}

| Stage | Shape | Values |
|---|---|---|
| file read from disk (`read_image`) | 2136 × 3216 × 3 | whole numbers 0–255 |
| black border cropped | 2136 × 2706 × 3 | 0–255 |
| padded to a square | 2706 × 2706 × 3 | 0–255 |
| resized and masked ("plain", stored in the cache) | 384 × 384 × 3 | 0–255 |
| colour-normalised (Ben Graham) | 384 × 384 × 3 | 0–255 |
| tensor for the network | 3 × 384 × 384 | real numbers, roughly −2 to +2.6 |
| a batch of training images | 24 × 3 × 384 × 384 | real numbers |

The last two rows move the channel axis to the front (PyTorch's convention: channels, height, width) and stack images into a **batch**, a group processed together.

## Image preprocessing

**Preprocessing** is every fixed transformation applied to *every* image, in training and at prediction time alike, so the network always sees images in the same standard form. The project's pipeline is defined once in `drscreen/preprocess.py` and reused everywhere (training cache, quality gate, application), which guarantees that training and use see identical transformations.

{{fig:preprocess_steps.png|The project's actual preprocessing applied to a real APTOS image (dataset grade 2). The final panel shows the green channel of the network's input after scaling; red means above the ImageNet average, blue below.|100|prep}}

### Cropping the black border

- **What:** find the pixels that belong to the retina (brighter than a small threshold after a light blur, cleaned with a morphological opening) and crop the image to their bounding box (`retina_mask`, `crop_black_borders`).
- **Why:** the black surround carries no information, and its size varies between cameras.
- **Trade-off:** a very dark image can make the retina hard to separate from the background; if less than 5% of the image looks like retina, the image is left uncropped rather than cropped badly.

### Padding to a square and resizing

- **What:** add black borders to make the image square (`pad_to_square`), then resize to 384 × 384 pixels. Shrinking uses area averaging, enlarging uses cubic interpolation.
- **Why:** the network needs a fixed size; padding instead of stretching keeps the retina round.
- **Trade-off:** shrinking a 3,000-pixel image to 384 pixels loses detail. A microaneurysm that is 10 pixels wide in the original becomes one or two pixels. The project's classifiers use 384 pixels (the B3 model 448); the lesion network works on a larger 1,280-pixel canvas because its targets are tiny.

### Circular field-of-view mask

- **What:** set every pixel outside a centred circle (98% of the image width) to black (`circle_mask`).
- **Why:** after resizing, the corners still contain bits of the camera's edge or padding; a clean, identical circle for every image removes an irrelevant cue the network could otherwise latch on to.

### Colour normalisation (Ben Graham's method)

The most important colour step subtracts a heavily blurred copy of the image from the image itself. This technique was popularised by the winner of the 2015 Kaggle DR competition [11]:

$$ I_{\mathrm{out}} = 4\,I - 4\,G_{\sigma} * I + 128 $$

Here $I$ is the image, $G_{\sigma} * I$ is the image blurred with a Gaussian filter of width $\sigma$ (the project uses $\sigma$ = image size / 30, about 13 pixels at 384), and 128 shifts the result to mid-grey. The blurred copy contains only slow variations: overall colour and uneven lighting. Subtracting it removes them and keeps the fine detail, such as vessels, lesions and edges, amplified four times. Pixels outside a slightly smaller circle (94%) are set to mid-grey (`ben_graham`).

- **Why:** photographs from different cameras and lighting conditions become much more alike, so the network does not waste capacity learning colour differences that have nothing to do with disease.
- **Trade-off:** the output looks unnatural to a human, and halos appear around very bright structures such as the optic disc. Strong noise is amplified too.

### Scaling to the statistics the network was pre-trained with

Finally each value is divided by 255 and standardised channel by channel with the mean and standard deviation of the ImageNet photographs the network was originally trained on (`to_tensor_array`):

$$ x_c = \frac{v_c / 255 - \mu_c}{s_c}, \qquad \mu = (0.485,\ 0.456,\ 0.406), \quad s = (0.229,\ 0.224,\ 0.225) $$

for channel $c$ (red, green, blue), pixel value $v_c$, ImageNet mean $\mu_c$ and standard deviation $s_c$. For example a green value of 200 becomes $(200/255 - 0.456)/0.224 \approx 1.47$. Pre-trained networks expect inputs on this scale; feeding them raw 0–255 values would make their learned filters respond wrongly.

::: warning Preprocessing is not an enhancement for the human eye
CLAHE contrast enhancement (a different technique, [42]) exists in the code but is applied only when the quality gate flags an image as dark or low-contrast, and only before analysis by the model. Neither CLAHE nor the Ben Graham step makes the image "more correct"; they make it more uniform for the model.
:::

## Data augmentation

### Why show the network altered copies?

::: define Augmentation
**Data augmentation** creates random, label-preserving variations of each training image every time it is used: rotated, flipped, slightly zoomed, brighter or darker. The network then has to learn features that survive these changes instead of memorising the exact pixels of each training photograph.
:::

A fundus photograph has no natural "up": the retina can be rotated and mirrored without changing its grade, and real cameras vary in brightness. Augmentation exploits this. It is applied **only during training**. Validation, test and real predictions always use the plain preprocessed image, because measurements must be repeatable and must reflect real inputs.

{{fig:augmentation.png|Augmentation applied by the project's own code (`augment_plain`) to a real image. Top: four random standard augmentations. Bottom: examples of the acquisition defects added only when training the robustness-augmented model, with parameters inside the ranges used in training.|100|aug}}

{{tab:aug|The augmentations implemented in `drscreen/datasets.py` (applied to training images only).}}

| Transformation | Range | Used for |
|---|---|---|
| rotation | any angle, 0–360° | all classifiers |
| zoom | ×0.92 to ×1.08 | all classifiers |
| horizontal / vertical flip | each with probability 0.5 | all classifiers |
| contrast, brightness, gamma | ×0.85–1.15; ±12 levels; gamma 0.85–1.2 | all classifiers |
| Gaussian blur | probability 0.25, σ 0.5–4 pixels | robust models only |
| JPEG re-compression | probability 0.30, quality 8–60 | robust models only |
| dark / low-contrast / washed-out | probability 0.30 | robust models only |
| over-exposure | probability 0.15 | robust models only |

::: warning Augmentations must not change the medical meaning
A change that could alter the grade is not allowed. Heavy blur can erase microaneurysms, and colour shifts can make haemorrhages look like vessels. That is why the standard ranges are mild, and why the stronger "defect" augmentations were tested as a separate model with a pre-set comparison (Part VI) rather than switched on silently. Flips mirror a right eye into something that looks like a left eye; this is harmless for the overall grade, but the lesion module's quadrant labels (nasal, temporal) are computed on unflipped images only.
:::

## The dataset and how it is split

### APTOS 2019

The classifier is trained on the labelled training images of the **APTOS 2019 Blindness Detection** dataset [7], released for a Kaggle competition by the Asia Pacific Tele-Ophthalmology Society with images from Aravind Eye Hospital, India. Each image has one grade from 0 to 4. Access requires accepting the competition rules, which the project owner did, and the images were used only inside Kaggle notebooks for this research project.

- Labelled images found: **{{val:dataset.n_labelled_found|d}}**, all readable ({{val:dataset.n_unreadable|d}} unreadable).
- Grades over the whole set: {{val:dataset.grade_counts_total.0|d}} (grade 0), {{val:dataset.grade_counts_total.1|d}} (1), {{val:dataset.grade_counts_total.2|d}} (2), {{val:dataset.grade_counts_total.3|d}} (3), {{val:dataset.grade_counts_total.4|d}} (4). The classes are strongly **imbalanced**: almost half the images are grade 0, and grade 3 is about one in twenty.
- No patient identifiers, no left/right eye information, no lesion annotations and no image-quality labels are provided.

::: warning A real bug, caught by testing
The competition folder also contains `sample_submission.csv`, a template whose "diagnosis" column holds a constant placeholder. The first training run read it as labels (5,590 rows) and had to be discarded. The loader now ignores submission files and constant-label tables, and refuses to start unless exactly 3,662 labelled images are found. This is a good example of why a pipeline must check its inputs.
:::

### Training, validation and test sets

::: define Three splits, three jobs
- The **training set** is used to learn the network's weights.
- The **validation set** is used to make choices: which epoch's weights to keep, the temperature for calibration, and the decision threshold.
- The **test set** is used once, at the end, to measure performance. It played no part in any choice.

An analogy: training is studying from a textbook, validation is the practice exams you use to decide how to study, and the test is the real exam whose questions you have never seen. If you peek at the real exam while practising, your score no longer measures what you learned.
:::

**Data leakage** happens when information from the test set influences training or selection, which makes results look better than they will be on new data. In medical imaging a classic leak is the same patient appearing in both training and test sets: two photographs of one eye, or a patient's two eyes, are much more alike than photographs of different people, so a model can "recognise" the patient instead of the disease.

### What the project actually does

APTOS gives no patient identifiers, so a true patient-level split is impossible. The project does the next best thing (`drscreen/datasets.py`, `make_splits`):

1. It computes for every image an exact fingerprint (the MD5 hash of the file) and a perceptual fingerprint (a 16 × 16 *difference hash* that changes little when an image is resized or recompressed).
2. Images with the same MD5, or whose difference hashes differ in at most 6 of 256 bits, are put in the same **group** ({{val:dataset.n_duplicate_groups|d}} groups; {{val:dataset.n_images_in_merged_groups|d}} images belonged to groups of two or more).
3. A **stratified, group-aware split** (scikit-learn's `StratifiedGroupKFold`, seed 42) assigns whole groups to one split while keeping the grade proportions similar in each split.
4. An assertion stops the program if any group or file hash appears in two splits.

{{fig:split_flow.png|The real data-preparation workflow, with counts from the saved run.|100|splitflow}}

{{tab:splits|Images per split and grade in the saved run.}}

| Split | Images | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4 | Referable share |
|---|---|---|---|---|---|---|---|
| training | {{val:dataset.splits.train.n|d}} | {{val:dataset.splits.train.grade_counts.0|d}} | {{val:dataset.splits.train.grade_counts.1|d}} | {{val:dataset.splits.train.grade_counts.2|d}} | {{val:dataset.splits.train.grade_counts.3|d}} | {{val:dataset.splits.train.grade_counts.4|d}} | {{val:dataset.splits.train.referable_prevalence|.3f}} |
| validation | {{val:dataset.splits.val.n|d}} | {{val:dataset.splits.val.grade_counts.0|d}} | {{val:dataset.splits.val.grade_counts.1|d}} | {{val:dataset.splits.val.grade_counts.2|d}} | {{val:dataset.splits.val.grade_counts.3|d}} | {{val:dataset.splits.val.grade_counts.4|d}} | {{val:dataset.splits.val.referable_prevalence|.3f}} |
| test | {{val:dataset.splits.test.n|d}} | {{val:dataset.splits.test.grade_counts.0|d}} | {{val:dataset.splits.test.grade_counts.1|d}} | {{val:dataset.splits.test.grade_counts.2|d}} | {{val:dataset.splits.test.grade_counts.3|d}} | {{val:dataset.splits.test.grade_counts.4|d}} | {{val:dataset.splits.test.referable_prevalence|.3f}} |

{{fig:class_distribution.png|Images per grade in each split (figure produced by the training run). The imbalance is the same in every split, by design.|72|classdist}}

::: warning What the split cannot rule out
Duplicate grouping catches the *same photograph* appearing twice, not *two different photographs of the same patient*. If APTOS contains both eyes of some patients, they may sit on different sides of the split, and the internal test results may be somewhat optimistic. The EyePACS experiment, where patient identifiers exist, splits strictly by patient.
:::

::: project Reproducible inputs
Every image is preprocessed once and stored in a 384 × 384 cache (a memory-mapped NumPy file) together with its hash and original size; the split is saved as `artifacts/split_manifest.csv`. Every later model (standard, robust, B3, EyePACS) re-creates the same split from the same code and seed, and comparison scripts refuse to compare models trained on different splits.
:::

### Other datasets used

{{tab:datasets|All datasets used by the project and their role.}}

| Dataset | Content | Role in the project |
|---|---|---|
| APTOS 2019 [7] | 3,662 graded photographs (India) | training, validation and internal test of the classifiers |
| IDRiD [8] (CC BY 4.0, third-party Kaggle copy) | 81 images with pixel lesion masks; disc/fovea centres; ~516 graded images | training and testing the lesion network and locator; **independent external check** of the classifier on its grading test images |
| e-ophtha [9] | microaneurysm and exudate masks, healthy images (France) | extra training data for lesion model v2 |
| EyePACS (Kaggle 2015) [10] | 35,126 graded photographs from US screening; patient IDs in file names | extra training data for the EyePACS experiment; held-out patients for testing |
