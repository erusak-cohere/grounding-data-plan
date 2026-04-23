# Visual Grounding Dataset Plan

## Existing Grounding-Related Data in the Mix

Grounding and spatial-coordinate datasets collectively represent **~3.2%** of the current training weight. This is low compared to SotA models (Qwen3-VL, STEP3-VL) which dedicate significant portions to grounding.

### By dataset family (% of grounding data, i.e. grounding = 100%):

**PixMo point annotations -- 21.2%** of grounding:
- `pixmo_point_expl` (11.7%)
- `openbee_honey_general_pixmo_points_explanations_v1_en` (9.5%)
- Note: other PixMo datasets (pixmo_ama, pixmo_docs, pixmo_cap_*) are general VQA/caption, not grounding

**CLEVR + Super-CLEVR (synthetic 3D scenes, counting) -- 20.9%** of grounding:
- `openbee_honey_grounding_and_counting_clevr_v1_en` (19.8%)
- + 3 smaller variants: super_clevr (0.5%), clevr_change (0.3%), clevr_math (0.2%)

**LVIS (fine-grained detection instructions) -- 19.1%** of grounding:
- `openbee_honey_general_lvis_instructv4_v1_en` (17.8%)
- `mammoth_lvis_instruct4v_220k` (1.3%)

**VisualGenome (scene graphs, regions) -- 10.6%** of grounding:
- `openbee_honey_grounding_and_counting_visualgenome_v1_en` (7.7%)
- `mammoth_visualgenome_llava_next` (2.9%)

**Other grounding/counting (TQA, IconQA, MovieNet, MathV360K) -- 7.5%** of grounding

**Localized Narratives (grounded captions with mouse traces) -- 6.5%** of grounding:
- `the_cauldron_localized_narratives` (6.5%)

**ShareGPT4V-SAM (grounded captions + SAM masks) -- 4.9%** of grounding

**TallyQA (counting on natural images) -- 3.5%** of grounding

**OCR bbox (text detection boxes) -- 3.0%** of grounding:
- ctw_bbox, hiertext_bbox, rects_cropped, rects_full, mammoth_rects, openbee_ocr_rects

**Visual7W (pointing QA) -- 2.7%** of grounding

**Objects365 (object detection, 365 categories) -- 0.3%** of grounding (!!):
- Only 296 packed rows

---

## Key Gap Analysis

Compared to SotA (Qwen3-VL, STEP3-VL):

1. **No RefCOCO/+/g** -- the classic referring expression benchmark family, used by Qwen3-VL for training
2. **Objects365 is negligible** -- 296 packed rows vs. millions of annotations in SotA recipes
3. **No Merlin** -- used by STEP3-VL for both bbox and point grounding (dataset not publicly available)
4. **No OpenImages detection** -- used by both Qwen3-VL and STEP3-VL at scale
5. **No reasoning traces for grounding** -- SotA models (Long Grounded Thoughts, Perception-R1) show CoT improves spatial understanding
6. **Total grounding is ~3.2%** -- likely insufficient for competitive spatial understanding

---

## Datasets to Download and Ingest

All new datasets will use **[0, 1000] normalized coordinates** (Qwen3-VL convention, see below). Existing coordinate datasets will also be reprocessed to this format.

### Batch 1: Core Grounding

| Dataset | Description | Size | Source |
|---------|-------------|------|--------|
| **RefCOCO** | Referring expression comprehension on COCO images. Short expressions (avg 3.5 words), allows location words. 14% label error rate. | ~120K expressions, ~20K images |https://github.com/lichengunc/refer |
| **RefCOCO+** | Same as RefCOCO but location words forbidden (appearance-only). 24% label error rate. | ~120K expressions |https://github.com/lichengunc/refer|
| **RefCOCOg** | Longer, more complex referring expressions (avg 8.4 words). 5% error rate. | ~85K expressions |https://github.com/lichengunc/refer|
| **Ref-L4 quality flags** | Reviewed annotations marking errors in RefCOCO/+/g with `caption_quality=0`. Use to filter bad labels. Ref-L4's own data (45K annotations on Objects365 images) is eval-only. | Annotations only | `JierunChen/Ref-L4` |
| **Objects365** (scale-up) | Large-scale object detection across 365 categories. Current version has only 296 packed rows. Download 200K subset with category balancing. | 200K subset (from 1.74M train total) | `jxu124/objects365` |
| **Long Grounded Thoughts (stage1)** | MCQ visual reasoning with long CoT traces. Synthesized from DOCCI images + Grounded-SAM bboxes. Key "with-reasoning" grounding dataset. | 753K SFT examples, 15K DOCCI images | `nvidia/nemotron-research-LGT` |
| **VisionFoundry-10K** | Synthetic perception triples covering 10 spatial tasks (orientation, viewpoint, depth, spatial relations). Fully synthetic T2I images. | 10K samples | `zlab-princeton/VisionFoundry-10K` |
| **FSC147** | Few-shot counting dataset (147 object categories). Used by STEP3-VL for counting.  | 6,135 images | [Kaggle](https://www.kaggle.com/datasets/xuncngng/fsc147-0) |
| **OODVQA** (full) | OOD VQA + counting on distribution-shifted images (unusual styles, sketches). From "How Many Unicorns" (ECCV 2024). We have ~half already via cambrian. | ~8.5K rows total | [UCSC-VLAA](https://github.com/UCSC-VLAA/vllm-safety-benchmark) |
| **OpenImages V7 Detection** (100K subset) | Category-balanced subset of OpenImages detection. 600 object classes. Used by Qwen3-VL and STEP3-VL. | 100K images (from 1.9M total) | `vikhyatk/openimages-bbox` |

### Batch 2 (deferred)

| Dataset | Description | Size | Source |
|---------|-------------|------|--------|
| **OpenImages Localized Narratives** | **Deferred to P2 (recapping).** We already have 200K rows via `the_cauldron_localized_narratives` (plain captions, no spatial data). The full dataset (671K) adds mouse traces that spatially ground each word to image regions -- useful as metadata for recapping. | 671K narratives | `HuggingFaceM4/LocalizedNarratives` |

### Dropped / Deferred

| Dataset | Reason |
|---------|--------|
| **Merlin** | Dataset never publicly released. Open GitHub issues requesting data since Mar 2024 remain unanswered. STEP3-VL likely used an internal copy. |
| **GroundUI-18K** | Test set only. GUI grounding deferred to future batch. |
| **Long Grounded Thoughts stage2 / DPO** | Ingest stage1 first, evaluate quality. |
| **Recapping grounding datasets with Qwen3.6** | P2 priority. Currently recapping Openbee; will extend to grounding datasets later. |

---

## Coordinate Reprocessing

We should switch all bounding box / point coordinates to **integer [0, 1000] format** (i.e. coordinates normalized to 0-999 integers). Evidence supporting this choice:

- **Qwen3-VL** (arXiv:2511.21631) uses [0, 1000] range for all grounding coordinates.
- **ChartPoint** (arXiv:2512.00305, ICCV 2025) ran an explicit ablation (Table 7): integer [0-999] coordinates outperform normalized [0-1] decimals by +1.16% on ChartQA. The paper notes that Qwen's tokenizer splits decimals into three-digit segments, increasing token-level training difficulty for floating-point formats.

After inspecting text samples from all candidate datasets (via uncompressed JSONL shards), only **3 datasets** in the current mix actually contain coordinates in the text. All other grounding/counting datasets (CLEVR, TallyQA, LVIS, Objects365, rects, pixmo_points, etc.) encode tasks as pure text Q&A without explicit spatial coordinates.

**Datasets with coordinates (must reprocess to [0, 1000]):**

- `openbee_honey_grounding_and_counting_visualgenome_v1_en` -- [0-1] floats, e.g. `[0.708,0.787,0.728,0.838]`
- `hiertext_bbox_v1` -- [0-1] floats in JSON, e.g. `"bbox": [0.07, 0.22, 0.61, 0.27]`
- `openbee_honey_general_sharegpt4v_sam_v1_en` -- **mixed formats!** [0-1] floats for input regions (e.g. `[0.65, 0.19, 0.83, 0.69]`) AND raw pixel integers for bbox output (e.g. `"bbox_2d": [59, 118, 361, 478]`)

**Issue: the current mix has inconsistent coordinate formats.** `sharegpt4v_sam` uses both [0-1] normalized floats and raw pixel-value integers within the same conversation turns. This means the model is trained on conflicting coordinate conventions, which likely hurts grounding performance.

---

## Execution Order

### Batch 1: All core grounding datasets (parallel)
- RefCOCO / RefCOCO+ / RefCOCOg (+ Ref-L4 quality flags)
- Objects365 200K subset
- Long Grounded Thoughts stage1 (753K)
- VisionFoundry-10K
- FSC147
- OODVQA (full)
- OpenImages V7 Detection 100K subset
- Reprocess 3 existing datasets to [0,1000] coords

### Batch 2 (deferred)
- OpenImages Localized Narratives -- P2 (recapping with mouse traces)
- Long Grounded Thoughts stage2 / DPO
- Recapping grounding datasets with Qwen3.6

---

## Implementation

All ingestion goes through the `data_acquisition` pipeline in a dedicated git worktree. Each dataset gets a `process.py` implementing `BaseDatasetAcquisition.transform_row()`, outputting the `multimodal_ift` schema (images + conversations) to GCS as parquet.
