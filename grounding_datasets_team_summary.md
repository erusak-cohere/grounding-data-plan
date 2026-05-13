# Visual Grounding Dataset Plan -- Team Summary

## Strategy: Nemotron as Core Grounding Dataset

With the Nemotron OI BBox v2 recaptioning pipeline (see `nemotron_oi_bbox_recaptioning_full.md`), we gain **1.66M high-quality grounding samples** from OpenImages V7 covering 600+ categories across four task types:

- **Task A** -- Grounding (text -> bbox)
- **Task B** -- Recognition (bbox -> text)
- **Task C** -- Counting (direct / box-based / verification)
- **Task D** -- Relational (spatial reasoning between objects)

All with reasoning traces (`<START_THINKING>...<END_THINKING>`) and `<box>[x1, y1, x2, y2]</box>` normalized [0, 1000] coordinates.

This subsumes the majority of existing grounding datasets in the mix and eliminates most planned new ingestions. The plan below retains only datasets that provide **unique modalities** Nemotron does not cover.

---

## Existing Grounding Data in the Mix

Grounding and spatial-coordinate datasets collectively represent **~3.2%** of the current training weight (balanced config). Below is the status of each family after reassessment.

### KEEP -- Unique modalities not covered by Nemotron

**PixMo point annotations -- 0.668% of total weight:**
- `pixmo_point_expl` (0.369%)
- `openbee_honey_general_pixmo_points_explanations_v1_en` (0.299%)
- **Why keep:** Point annotations are a fundamentally different spatial modality (pointing to locations vs. drawing boxes). Nemotron has zero point data.

**LVIS (fine-grained detection instructions) -- 0.602% of total weight:**
- `openbee_honey_general_lvis_instructv4_v1_en` (0.561%)
- `mammoth_lvis_instruct4v_220k` (0.041%) -- remove this duplicate, keep openbee only.
- **Why keep:** 1200+ fine-grained categories (double OpenImages' 600+), with detailed instruction format. Provides category granularity Nemotron may not cover. Reassess after ablation results.

**OCR BBox / text detection -- 0.091% of total weight:**
- `ctw_bbox_v1`, `hiertext_bbox_v1`, `rects_cropped`, `rects_full`, `mammoth_rects_train`
- needs bbox conversion from [0-1] floats to [0, 1000] integers for `hiertext_bbox_v1` and `ctw_bbox_v1`.
- **Why keep:** Text detection in images is a completely different task from object detection. Nemotron has no OCR data.

**Localized Narratives -- 0.205% of total weight:**
- `the_cauldron_localized_narratives` (0.205%)
- **Why keep:** Grounded captions linking narrative text to image regions via mouse traces. Unique annotation style not covered by Nemotron.

### DROP -- Subsumed by Nemotron

**CLEVR + Super-CLEVR (synthetic) -- 1.336% of total weight (12 datasets):**
- All CLEVR variants: `openbee_honey_grounding_and_counting_clevr_v1_en`, `clevr_math_train`, `cambrian_10m_clevr`, `mammoth_clevr_700k`, `super_clevr_train`, `the_cauldron_clevr`, `openbee_honey_grounding_and_counting_super_clevr_v1_en`, `mammoth_super_clevr`, `openbee_honey_grounding_and_counting_clevr_math_v1_en`, `mammoth_clevr_math`, `mammoth_internvl2_llama3_super_clevr`, `openbee_honey_grounding_and_counting_clevr_change_v1_en`
- **Why drop:** Synthetic 3D rendered geometric shapes. Nemotron's counting tasks (C1/C2/C3) on real photos provide more relevant counting capability. Balanced-weight ablation already showed no regression when removing CLEVR duplicates.

**VisualGenome -- 0.333% of total weight (2 datasets):**
- `openbee_honey_grounding_and_counting_visualgenome_v1_en`, `mammoth_visualgenome_llava_next`
- **Why drop:** Known repetition bug in turns. Uses [0-1] floats requiring reprocessing. Nemotron's consolidated scene annotations + Task B (recognition) cover region descriptions at 200x the scale.

**TallyQA -- 0.295% of total weight (4 datasets):**
- `cambrian_10m_tallyqa`, `openbee_honey_grounding_and_counting_tallyqa_v1_en`, `the_cauldron_tallyqa`, `mammoth_tallyqa`
- **Why drop:** Pure counting on natural images. Nemotron's Task C (3 sub-formulations: direct, box-based, verification) is a direct and much larger replacement.

**ShareGPT4V-SAM -- 0.154% of total weight (2 datasets):**
- `openbee_honey_general_sharegpt4v_sam_v1_en`, `mammoth_sharegpt4v_sam`
- **Why drop:** Mixed coordinate formats ([0-1] floats AND raw pixel integers in same turns) actively confuse the model. Nemotron provides clean, consistent grounding data.

**IconQA -- 0.197% of total weight (5 datasets):**
- `openbee_honey_general_iconqa_v1_en`, `openbee_honey_grounding_and_counting_iconqa_v1_en`, `the_cauldron_iconqa`, `mammoth_iconqa`, `mammoth_internvl2_llama3_iconqa`
- **Why drop:** Abstract icon-based diagrams, not natural images. Does not contribute to spatial grounding on real photos.

**Visual7W -- 0.085% of total weight (3 datasets):**
- `openbee_honey_general_visual7w_v1_en`, `mammoth_visual7w`, `the_cauldron_visual7w`
- **Why drop:** Pointing QA, small scale. Covered by Nemotron's Task B (recognition) and Task D (relational).

**Objects365 -- 0.008% of total weight (1 dataset):**
- `openbee_honey_general_objects365_v1_en` (296 packed rows)
- **Why drop:** Negligible scale. Objects365's 365 categories are largely a subset of OpenImages' 600+. Nemotron covers object detection at 5000x the scale.

**Other tiny datasets (TQA, MathV360K, MovieNet, Other) -- 0.157% of total weight:**
- `openbee_honey_grounding_and_counting_tqa_v1_en`, `openbee_honey_grounding_and_counting_mathv360k_vqa_as_v1_en`, `openbee_honey_grounding_and_counting_movienet_v1_en`, `openbee_honey_grounding_and_counting_other_v1_en`
- **Why drop:** Combined <1K packed rows. Marginal value given Nemotron's scale.

---

## Gap Analysis (updated)

Previous gaps compared to SotA (Qwen3-VL, STEP3-VL) and how Nemotron addresses them:

| Gap | Status |
|-----|--------|
| No RefCOCO/+/g | **Closed by Nemotron.** 1.66M grounding + recognition samples subsume RefCOCO's ~70K rows. No separate ingestion needed. |
| Objects365 is negligible (296 rows) | **Closed by Nemotron.** OpenImages V7 600+ categories at 1.66M scale. |
| No OpenImages detection | **Closed by Nemotron.** Nemotron IS OpenImages V7 detection data. |
| No reasoning traces for grounding | **Closed by Nemotron.** All recaptioned 1.66M samples will include `<START_THINKING>...<END_THINKING>` reasoning. |
| Total grounding is ~3.2% | **Will increase** with Nemotron added to the mix. Exact weight TBD from ablation results. |
| No Merlin | **Cannot address.** Dataset never publicly released. |

**Remaining gaps:**
- No GUI grounding (GroundUI-18K is eval-only, deferred).
- Point grounding limited to PixMo (no Merlin).

---

## Datasets to Ingest

### Active
`
| Dataset | Description | Size | Status |
|---------|-------------|------|--------|
| **Nemotron OI BBox v2 (recaptioned)** | Core grounding dataset. 4 task types, reasoning traces, `<box>[x1,y1,x2,y2]</box>` format. See `nemotron_oi_bbox_recaptioning_full.md`. | ~1.66M samples | Recaptioning pipeline: TODO |
| **Long Grounded Thoughts (stage1)** | MCQ visual reasoning with long CoT traces. Synthesized from DOCCI images + Grounded-SAM bboxes. Unique reasoning-heavy grounding format. | 753K SFT examples, 15K DOCCI images | TODO |
| **VisionFoundry-10K** | Synthetic perception triples covering 10 spatial tasks (orientation, viewpoint, depth, spatial relations). Complements Nemotron with tasks it doesn't cover. | 10K samples | Ablation running |

### Dropped (subsumed by Nemotron)

| Dataset | Reason |
|---------|--------|
| **RefCOCO/+/g** | 70K rows vs. Nemotron's 1.66M. Grounding + recognition tasks fully subsumed. Avoiding evaluation contamination is an additional benefit. |
| **Objects365 (200K expansion)** | Objects365's 365 categories are a subset of OpenImages' 600+. Nemotron covers this at 8x the planned scale. |
| **OpenImages V7 Detection (100K)** | Nemotron IS OpenImages data. Completely redundant. |
| **FSC147** | Few-shot counting subsumed by Nemotron's counting tasks (C1/C2/C3) at much larger scale. |
| **OODVQA** | Low priority. Small scale (~8.5K rows). Could revisit if OOD robustness is a concern. |

### Deferred

| Dataset | Reason |
|---------|--------|
| **Merlin** | Never publicly released. |
| **GroundUI-18K** | Test set only. GUI grounding deferred. |
| **LGT stage2 / DPO** | Ingest stage1 first, evaluate quality. |
| **OpenImages Localized Narratives (full)** | Low priority. We already have 200K rows via `the_cauldron_localized_narratives`. |

---

## Coordinate Format

All bounding box coordinates use **integer [0, 1000] normalized format**, rendered as `<box>[x1, y1, x2, y2]</box>` (HTML-style box tags, no special tokenizer tokens). Evidence:

- **Qwen3-VL** (arXiv:2511.21631) uses [0, 1000] range for all grounding coordinates.
- **ChartPoint** (arXiv:2512.00305, ICCV 2025): integer [0-999] outperforms [0-1] decimals by +1.16% on ChartQA.
- **No tokenizer update needed:** `<box>` / `</box>` are standard text tokens handled natively by any tokenizer. This format is unambiguous (zero false positives in natural text) and trivially extractable via regex.

Datasets being kept (PixMo, LVIS, OCR BBox, Localized Narratives) encode tasks as pure text Q&A without explicit spatial coordinates in the text, so no coordinate reprocessing is needed for them. The only dataset that previously required coordinate reprocessing (`openbee_honey_grounding_and_counting_visualgenome_v1_en` and `openbee_honey_general_sharegpt4v_sam_v1_en`) are being dropped.

`hiertext_bbox_v1`, `ctw_bbox_v1` (OCR BBox, kept) use [0-1] floats in the text and should be reprocessed to [0, 1000] integers.

---

## Implementation

**Priority 1:** Nemotron OI BBox v2 recaptioning pipeline (see `nemotron_oi_bbox_recaptioning_full.md`). VisionFoundry-10K (ablation already running).

**Priority 2:** Long Grounded Thoughts stage1 ingestion.

**Cleanup:** Set weights of all dropped datasets to 0 in the training config. Redistribute freed weight (~2.5% of total) to Nemotron and remaining datasets based on ablation results.

**Ablations in progress:**
- VisionFoundry-10K: aggressive replace, conservative replace, append (configs ready)
- Nemotron (pre-recaptioning): aggressive replace, conservative replace (configs ready)

See `mm-configs-grounding-ablation/bls/vision/sft/` for all ablation `.run` and `_sweep.py` files.
