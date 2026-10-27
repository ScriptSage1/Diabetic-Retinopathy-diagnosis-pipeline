# Biological foundations

Before any algorithm: what the eye is, what a retinal photograph shows, how diabetes damages the retina, and what exactly the project tries to detect.

## The eye: an optical instrument and a sensor

### Two jobs: focusing light and sensing it

The eye does two different jobs. The front of the eye **focuses** light; the back of the eye **senses** it ({{ref:eye}}).

- The **cornea** is the clear, curved window at the front. Because light bends strongly when it passes from air into the cornea, the cornea does most of the eye's focusing.
- The **iris** is the coloured ring of muscle behind the cornea; the **pupil** is the opening in its centre. The iris widens or narrows the pupil to control how much light enters.
- The **lens**, just behind the iris, fine-tunes the focus by changing shape (this is how we switch between near and far objects).
- The **vitreous humour** is the clear gel that fills the large chamber of the eye and keeps its shape.
- The **retina** is a thin layer of nerve tissue that lines the inside of the back of the eye. It is the sensor: it turns light into nerve signals.
- The **optic nerve** carries those signals to the brain. It leaves the eye at the **optic disc**, which is also where the retina's main blood vessels enter and leave.
- The **macula** is a small area of the retina near the centre of the back of the eye, responsible for detailed central vision (reading, recognising faces). At its centre is a small pit, the **fovea**, where vision is sharpest.
- The **sclera** is the tough white outer wall.

{{fig:eye_anatomy.png|Schematic cross-section of the eye (original diagram, not to scale). Light is focused by the cornea and lens onto the retina; the sharpest image falls on the fovea. Signals leave through the optic nerve at the optic disc.|92|eye}}

::: why Why a computer scientist needs this
Every structure in this diagram appears in a retinal photograph or affects how that photograph looks. The optic disc is the brightest round object in the image, the macula is the darker area beside it, and the blood vessels radiate from the disc. The lesion part of the project explicitly locates the disc and the fovea, because the clinical grading rules and the "near the macula" warning depend on where findings lie relative to them.
:::

### The retina, rods and cones, and its blood supply

Light that reaches the retina passes through several transparent layers of nerve cells before it is absorbed by the **photoreceptors** at the back ({{ref:layers}}). There are two kinds [5]:

- **Rods** are very sensitive and work in dim light, but do not distinguish colour. There are roughly a hundred million of them, mostly outside the centre.
- **Cones** work in brighter light, provide colour vision and fine detail, and are concentrated in the fovea.

A photoreceptor that absorbs light changes its electrical state; the signal is passed through **bipolar cells** to **ganglion cells**, whose long fibres (axons) run across the inner surface of the retina to the optic disc and become the optic nerve.

The retina is metabolically very active, so it needs a rich blood supply. Its **inner layers are fed by the retinal blood vessels**, which branch out from the optic disc across the inner surface; its outer layers, including the photoreceptors, are fed from behind by the **choroid**. The retinal vessels lie in front of the photoreceptors, so they are visible in a photograph taken through the pupil, and it is their damage that diabetic retinopathy is about.

{{fig:retina_layers.png|Simplified layers of the retina (original diagram). Light crosses the transparent inner layers to reach the photoreceptors; signals travel back to the ganglion cells and along their fibres to the optic nerve. Retinal blood vessels supply the inner retina.|92|layers}}

::: define Why the retina reveals disease
The retina is the only place in the body where small blood vessels and nerve tissue can be photographed directly and without surgery. Changes to those vessels, such as leaks, bleeds and closures, become visible as spots and patches in a photograph. That is what makes automated screening from photographs possible at all.
:::

## The fundus photograph

### What the camera captures

The **fundus** is the interior back surface of the eye: the retina, optic disc, macula and vessels. A **fundus camera** shines light through the pupil and photographs the fundus through the same opening, typically covering a field of about 30° to 50° around the posterior pole [6]. The result is a colour image in which the retina appears as a reddish-orange disc on a black background (the black is the area outside the camera's circular field of view).

{{fig:fundus_annotated.png|A real fundus photograph from the APTOS 2019 test split (dataset grade 0). Left: as stored. Right: after the project's centring step, with the optic disc and fovea as estimated by the project's own locator (described with the lesion pathway in Part III). The macula circle is an approximate guide, not a measured boundary.|96|fundus}}

In {{ref:fundus}} you can find, from left to right: the bright **optic disc**, with the large vessels leaving it; the vessels arching above and below towards the centre; and the darker **macula** with the **fovea** at its middle. This particular eye has no visible retinopathy according to the dataset's label.

### Why photographs are hard for a computer

Two photographs of healthy eyes can look very different. The main sources of variation are:

- **Illumination**: uneven lighting, a dark rim, or a bright reflection in the middle.
- **Colour**: different cameras, flash settings and pigmentation of the eye change the overall hue.
- **Focus and motion**: a blurred image hides small lesions.
- **Field of view and framing**: the retina may be off-centre, partly cut off, or a different size in the frame.
- **Resolution**: images in the project's datasets range from about 400 to more than 4,000 pixels wide.
- **Artefacts**: eyelashes, dust on the lens, compression blocks from JPEG.

Each of these is a reason for a processing step later in the guide: centring and cropping fix framing, resizing fixes resolution, colour normalisation reduces illumination and colour differences, the quality gate rejects unusable photographs, and augmentation teaches the model to tolerate the rest.

::: project In our project
A photograph enters the system as a file (PNG or JPEG) and is turned into an array of numbers by `drscreen.preprocess.read_image`, which also rejects missing, corrupt, too-small or single-colour files with a clear message. Part II shows exactly what those numbers are.
:::

## Diabetes and diabetic retinopathy

### From blood sugar to damaged vessels

**Diabetes mellitus** is a group of conditions in which the blood-glucose level stays too high, because the body produces too little insulin or responds to it poorly. Over years, persistently high glucose damages the walls of small blood vessels throughout the body, including the retina [2, 3]. In simplified terms:

1. The cells that support capillary walls are lost and the walls weaken; small bulges form (**microaneurysms**).
2. Weakened vessels **leak**: blood escapes (**haemorrhages**), and fluid and fats seep out and leave yellowish deposits (**hard exudates**). Leakage near the centre can swell the macula (**diabetic macular oedema**), which threatens central vision.
3. Some capillaries **close**. Parts of the retina are starved of oxygen (**ischaemia**), seen as fluffy pale patches (**cotton-wool spots**), abnormal vein shapes (**venous beading**) and abnormal small vessels inside the retina (**IRMA**).
4. The oxygen-starved retina releases growth signals that trigger fragile **new vessels** (**neovascularisation**). These can bleed into the vitreous or pull on the retina: this is **proliferative** diabetic retinopathy, which can cause severe loss of vision.

Diabetic retinopathy is common among people with diabetes, and its early stages are usually painless and unnoticed [2, 4]. Treatments such as laser therapy and injections of drugs that block the growth signal work best when the disease is caught in time, which is why people with diabetes are screened regularly.

::: warning Four things that are easy to confuse
A **disease process** (vessel damage from high glucose) is not the same as an **observable sign** (a haemorrhage in a photograph), a **model prediction** (a number produced by our software) or a **clinical diagnosis** (a qualified clinician's judgement using the examination, history and other tests). This project only ever produces model predictions from photographs.
:::

## Retinal lesions and clinical grading

### The lesions

{{tab:lesions|Retinal findings of diabetic retinopathy and what this project can do with each.}}

| Finding | What it is | Typical appearance in a photograph | In this project |
|---|---|---|---|
| Microaneurysm (MA) | small bulge in a capillary wall; usually the earliest visible sign | tiny round red dot | segmented by the lesion network |
| Haemorrhage (HE) | blood leaked into the retina | red dots, blots or flame shapes, larger than MAs | segmented |
| Hard exudate (EX) | lipid and protein deposits left by leaking vessels | sharp-edged yellow-white spots, often in clusters or rings | segmented |
| Cotton-wool spot (soft exudate, SE) | swelling of nerve fibres from local ischaemia | fluffy, pale, blurred-edged patches | segmented |
| Venous beading | veins with alternating swelling and narrowing | sausage-like veins | **not detected** (background only) |
| IRMA | abnormal small vessels within the retina | fine, irregular red vessels | **not detected** |
| Neovascularisation | fragile new vessels on the disc or retina | fine vessel networks, sometimes with bleeding | **not detected** |

{{fig:lesions_eophtha.png|Two lesion types in real photographs with the outlines drawn by the dataset's specialists (e-ophtha database [9], crops enlarged). Top: a microaneurysm, a dot only a few pixels wide even in a high-resolution image. Bottom: hard exudates.|56|eophtha}}

{{fig:lesion_idrid_test_examples.png|Four IDRiD test images [8]: the photograph, the specialists' lesion masks ("ground truth") and the project's lesion network's detections ("prediction"), drawn on the colour-normalised image. Red = microaneurysms, orange = haemorrhages, yellow = hard exudates, cyan = cotton-wool spots. Generated by the project's lesion evaluation; the detector finds most large lesions but misses or adds small ones.|78|idridles}}

Small lesions are genuinely hard to see. A microaneurysm can be smaller than a vessel's width ({{ref:eophtha}}), and a dark spot can be a lesion, a vessel crossing, a pigment variation or dust. Even experts disagree about individual lesions and about grades, which limits how "correct" any label in a dataset can be.

### The international grading scale

Clinicians summarise the severity of diabetic retinopathy with the **International Clinical Diabetic Retinopathy (ICDR) severity scale** [1]. It has five levels:

{{tab:icdr|The ICDR severity scale [1] and the project's referral definition.}}

| Grade | Name | Findings (simplified from [1]) | Project label |
|---|---|---|---|
| 0 | No apparent retinopathy | no abnormalities | non-referable |
| 1 | Mild non-proliferative DR (NPDR) | microaneurysms only | non-referable |
| 2 | Moderate NPDR | more than just microaneurysms, but less than severe | **referable** |
| 3 | Severe NPDR | any of: more than 20 intraretinal haemorrhages in each of 4 quadrants; definite venous beading in 2 or more quadrants; prominent IRMA in 1 or more quadrant; and no signs of proliferative DR (the "4-2-1 rule") | **referable** |
| 4 | Proliferative DR (PDR) | neovascularisation and/or vitreous or pre-retinal haemorrhage | **referable** |

The scale also has a separate part for diabetic macular oedema [1]. The project does not grade macular oedema; it only reports a *proxy* (hard exudates close to the estimated fovea) as a warning.

{{fig:grades_gallery.png|One real APTOS 2019 test image for each grade, as labelled in the dataset (after the project's preprocessing). Grades 2–4 count as referable. Differences between neighbouring grades can be subtle.|100|grades}}

::: project Where the grading scale lives in the code
The grade names and the referral cut-off are defined once, in `drscreen/metrics.py` (`GRADE_NAMES`, `REFERABLE_MIN_GRADE = 2`, `referable_labels`, `referable_prob`). The 4-2-1 rule and the other criteria are encoded in `drscreen/rules.py`, which marks each criterion *met*, *not met* or *not assessed*. The last applies to venous beading, IRMA and new vessels, because no detector for them exists.
:::

## The screening problem this project solves

**Screening** means testing many people who have no symptoms in order to find the few who need a specialist. **Diagnosis** means a clinician establishing what a particular patient has. A screening test is designed to sort people quickly and cheaply into "refer" and "re-screen later"; a qualified clinician then examines the referred people.

The computational question the project answers is therefore narrow and precise:

> Given one colour fundus photograph of one eye, estimate the probability of each ICDR grade, and decide whether the eye shows **referable diabetic retinopathy (grade 2 or higher)**, together with evidence that a human can check.

It is a **five-class image classification** problem (grades 0–4) from which a **binary screening decision** (referable or not) is derived. Around it are three supporting tasks: **image-quality assessment** (is the photograph usable?), **lesion segmentation** (where are the lesions?) and **rule-based grading** (what do the ICDR criteria say about those lesions?).

{{fig:problem_to_computation.png|How the biological problem becomes a computational one (original diagram).|100|p2c}}

::: takeaway Part I in three sentences
Diabetes damages the small vessels of the retina, producing visible lesions whose type and extent define five severity grades. A fundus camera photographs those lesions through the pupil, with large variation in light, colour, focus and framing. The project turns such a photograph into grade probabilities and a refer / do-not-refer recommendation, which is a screening aid and not a diagnosis.
:::
