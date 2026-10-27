# Biology & Clinical Topics for the DR Project

Everything biology-side you need to understand for this project. Each topic says **what to know** and **where it matters** (the roadmap module).
You don't need to become an ophthalmologist. The goal is to recognise what the model is looking at and why the grading rules exist.

---

## 1. Diabetes basics
- [ ] **What diabetes is:** the body can't control blood sugar (glucose) properly. Type 1: no insulin; Type 2: the body resists insulin. *(Module 0)*
- [ ] **HbA1c:** the standard blood test for long-term sugar control. Higher means more risk of eye damage. *(background for the report)*
- [ ] **Why diabetes hurts small blood vessels:** high sugar over years damages the walls of tiny vessels (capillaries) everywhere, especially in the eyes, kidneys and nerves. DR is one of these "microvascular" complications. *(Module 0, report intro)*
- [ ] **The India context:** large diabetic population, few eye specialists in rural areas, which is why automated screening is needed. *(report intro)*

## 2. Anatomy of the eye (just enough)
- [ ] **Basic path of light:** cornea → pupil → lens → vitreous (clear gel) → **retina**. Anything cloudy along this path (e.g. cataract) makes fundus photos hazy. *(Module 10)*
- [ ] **The retina:** the thin light-sensitive layer at the back of the eye, the "film" of the camera. DR damages the vessels that feed it. *(Module 0)*
- [ ] **Retina layers (simple version):** a surface nerve-fibre layer, deeper layers with photoreceptors, and blood vessels running through. Where bleeding happens changes how it looks (see haemorrhages below). *(Module 12)*

## 3. Reading a fundus photo
A fundus photo is a picture of the retina taken through the pupil.
- [ ] **Optic disc:** bright, yellowish round/oval spot where the optic nerve and blood vessels enter the eye. It's the brightest structure, so be careful: models can confuse it with exudates. *(Modules 0, 12)*
- [ ] **Macula:** darker area in the centre of the retina, roughly 2.5 disc-widths from the optic disc. Responsible for detailed central vision. *(Modules 0, 12, 14)*
- [ ] **Fovea:** the tiny centre of the macula, giving the sharpest vision. It has no blood vessels. Lesions near it are the most dangerous for sight. *(Modules 12, 14)*
- [ ] **Arteries vs veins:** arteries are thinner and lighter red; veins are thicker and darker. *(Modules 11, 14)*
- [ ] **Left vs right eye:** the optic disc sits on the nose side, so its position tells you which eye the image is from. Datasets like EyePACS include both eyes per patient. *(Module 7, patient-level splits)*
- [ ] **Quadrants:** the retina is divided into 4 quadrants around the optic disc/fovea. The severity rules count lesions per quadrant. *(Module 14)*
- [ ] **Why the green channel is best:** haemoglobin absorbs green light strongly, so vessels and red lesions show up dark with high contrast in the green channel. *(Modules 0, 1)*

## 4. How diabetic retinopathy develops
Follow the story; each step produces a lesion you'll detect:
- [ ] **1. Weak vessel walls** → tiny bulges form → **microaneurysms**. *(earliest sign)*
- [ ] **2. Vessels leak and burst** → **haemorrhages** (blood) and **hard exudates** (leaked fat/protein).
- [ ] **3. Vessels get blocked** → parts of the retina starve of oxygen (**ischaemia**) → **cotton-wool spots**, **venous beading**, **IRMA**.
- [ ] **4. The starving retina releases growth signals (VEGF)** → fragile **new vessels** grow (**neovascularisation**) → they bleed easily and can cause blindness.
- [ ] **Non-proliferative (NPDR) vs proliferative (PDR):** NPDR = steps 1–3, PDR = step 4 (new vessels). *(Module 0)*

## 5. The lesions: what they look like
This is the most important section. These are what your segmentation and rule engine detect.

| Lesion | What it is | What it looks like | Matters for |
|---|---|---|---|
| **Microaneurysm (MA)** | Tiny bulge in a capillary wall | Very small, sharp **red dot** (just a few pixels) | Earliest sign; MAs only = grade 1 (Modules 12, 13) |
| **Dot/blot haemorrhage** | Bleeding in deeper retinal layers | Red dots or blotches, bigger than MAs | Counted per quadrant for grade 3 (Module 14) |
| **Flame haemorrhage** | Bleeding in the surface nerve-fibre layer | Red streak, feathery, flame-shaped | Moderate+ disease (Module 12) |
| **Hard exudates** | Leaked fat/protein deposits | **Bright yellow-white** spots with sharp edges, often in clusters or rings | Near the fovea → risk of macular oedema (Modules 12, 14) |
| **Cotton-wool spots (soft exudates)** | Small areas of nerve-fibre damage from lack of blood | **Fluffy white** patches with blurry edges | Sign of ischaemia (Module 12) |
| **Venous beading** | Veins become irregular in width | Veins look like a string of sausages/beads | 4-2-1 rule (Module 14) |
| **IRMA** | Abnormal small vessels inside the retina | Thin, twisted, irregular vessels | 4-2-1 rule (Modules 13, 14) |
| **Neovascularisation (NVD/NVE)** | New fragile vessels on the disc (NVD) or elsewhere (NVE) | Fine, tangled, lacy vessels | Means PDR = grade 4 (Module 13) |
| **Vitreous/preretinal haemorrhage** | Bleeding from new vessels into the eye | Large dark red area, may hide the retina | Grade 4 (Module 14) |

- [ ] Can name each lesion when shown an example image
- [ ] Know which are **red lesions** (MA, haemorrhages) vs **bright lesions** (exudates, cotton-wool spots); the models treat them differently

## 6. The grading scale (ICDR)
- [ ] **Grade 0, no DR:** no abnormalities.
- [ ] **Grade 1, mild NPDR:** microaneurysms only.
- [ ] **Grade 2, moderate NPDR:** more than just microaneurysms, but less than severe.
- [ ] **Grade 3, severe NPDR:** any of the **4-2-1 rule**:
  - **4** quadrants with many (20+) haemorrhages
  - **2** quadrants with venous beading
  - **1** quadrant with prominent IRMA
  - …and no new vessels.
- [ ] **Grade 4, PDR:** neovascularisation and/or vitreous/preretinal haemorrhage.
- [ ] **Referable DR = grade 2 or higher.** This is the main yes/no question your system answers. *(Modules 3–7)*
- [ ] **Diabetic macular oedema (DME):** fluid/swelling at the macula, the most common cause of vision loss in diabetics. It can't be seen directly in a photo, but **hard exudates near the fovea** are the usual sign. Good to mention in the report even if you don't grade it separately. *(Modules 14, 18)*

## 7. Fundus photography and image quality
- [ ] **Fundus cameras:** desktop clinic cameras vs portable/handheld/smartphone cameras (cheaper, used in rural camps, lower quality). *(Module 10)*
- [ ] **Dilated vs undilated pupils:** eye drops widen the pupil for a better photo. Undilated photos are often darker with dark edges. *(Module 10)*
- [ ] **Common causes of bad images:** cataract (haze), small pupil (dark/vignetted edges), blinking/eyelashes, patient movement (blur), wrong camera alignment (optic disc or macula cut off), flash reflections. *(Module 10)*
- [ ] **Field of view:** a standard photo covers ~45° of the retina and should show both the optic disc and the macula. *(Module 10)*

## 8. Screening and clinical workflow
- [ ] **Screening vs diagnosis:** screening sorts many people quickly into "refer" or "re-screen later"; the specialist then diagnoses. Your system is a screening tool. *(Modules 5, 16)*
- [ ] **Why sensitivity comes first:** a missed referable case can mean preventable blindness; a false alarm only costs a specialist visit. *(Module 5)*
- [ ] **Referral urgency (simple version):** no/mild DR → re-screen in about a year; moderate → refer; severe/PDR → refer urgently. Use this in your report's recommendations. *(Module 16)*
- [ ] **Treatments exist (just know their names):** laser treatment, anti-VEGF eye injections, surgery (vitrectomy). This is why early detection prevents vision loss. *(report intro)*
- [ ] **Human-in-the-loop:** an ophthalmologist checks flagged or uncertain AI results; your report must let them verify a case quickly. *(Modules 9, 16, 17)*

---

## Good places to read more
- **EyeWiki (American Academy of Ophthalmology):** the *Diabetic Retinopathy* article, clear and well illustrated.
- **ICO Guidelines for Diabetic Eye Care** (International Council of Ophthalmology): the referral rules and grading in plain form.
- **Wilkinson et al. 2003** (*Ophthalmology*): the original paper defining the international DR severity scale (cite it in your report).
- **Practice:** open IDRiD images alongside their lesion masks. Seeing each lesion highlighted is the fastest way to learn to spot them.
