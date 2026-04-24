# Grounding Datasets in the Training Mix

**Total grounding weight (base):** 0.041490 (4.15% of all data)

**Total grounding weight (upweight):** 0.088680 (7.99% of all data)

**Training config:** 8000 steps x 256 batch = 2,048,000 total sample draws

---

## CLEVR / Super-CLEVR (synthetic counting/3D)

Combined base weight: 0.013360 (1.336% of total)

| Dataset | Base Weight | Upweight | Changed | Packed Rows | Base Epochs | Upweight Epochs |
|---------|-----------|----------|---------|-------------|-------------|----------------|
| `openbee_honey_grounding_and_counting_clevr_v1_en` | 0.00624716 | 0.00312358 | **YES** | 22,844 | 0.5601 | 0.2800 |
| `clevr_math_train` | 0.00393962 | 0.00000000 | **YES** | 22,752 | 0.3546 | 0.0000 |
| `cambrian_10m_clevr` | 0.00127667 | 0.00000000 | **YES** | 29,510 | 0.0886 | 0.0000 |
| `mammoth_clevr_700k` | 0.00082550 | 0.00000000 | **YES** | 26,434 | 0.0640 | 0.0000 |
| `super_clevr_train` | 0.00045754 | 0.00000000 | **YES** | 2,867 | 0.3268 | 0.0000 |
| `the_cauldron_clevr` | 0.00017814 | 0.00000000 | **YES** | 6,394 | 0.0571 | 0.0000 |
| `openbee_honey_grounding_and_counting_super_clevr_v1_en` | 0.00016709 | 0.00008355 | **YES** | 602 | 0.5684 | 0.2842 |
| `openbee_honey_grounding_and_counting_clevr_change_v1_en` | 0.00010482 | 0.00005241 | **YES** | 368 | 0.5833 | 0.2917 |
| `mammoth_super_clevr` | 0.00006136 | 0.00000000 | **YES** | 1,851 | 0.0679 | 0.0000 |
| `openbee_honey_grounding_and_counting_clevr_math_v1_en` | 0.00006002 | 0.00003001 | **YES** | 224 | 0.5487 | 0.2744 |
| `mammoth_internvl2_llama3_super_clevr` | 0.00002624 | 0.00000000 | **YES** | 790 | 0.0680 | 0.0000 |
| `mammoth_clevr_math` | 0.00001563 | 0.00000000 | **YES** | 507 | 0.0631 | 0.0000 |

## PixMo Points (point annotations)

Combined base weight: 0.006684 (0.668% of total)

Note: pixmo_point_expl contains pointing data in format: <points points="(21.5, 10.6), (21.3, 12.8), (21, 15.7), (23.2, 17.8), (22, 20), (22, 22.2), (22.9, 24.8), (22.6, 27.5)">, i.e. yet a different format from [0,1000] or the floating point format. `openbee_honey_general_pixmo_points_explanations_v1_en` does not seem to contain pointing data.

| Dataset | Base Weight | Upweight | Changed | Packed Rows | Base Epochs | Upweight Epochs |
|---------|-----------|----------|---------|-------------|-------------|----------------|
| `pixmo_point_expl` | 0.00368903 | 0.02213415 | **YES** | 17,941 | 0.4211 | 2.5267 |
| `openbee_honey_general_pixmo_points_explanations_v1_en` | 0.00299467 | 0.01796802 | **YES** | 10,798 | 0.5680 | 3.4079 |

## LVIS (fine-grained detection)

Combined base weight: 0.006020 (0.602% of total)

Note: `openbee_honey_general_lvis_instructv4_v1_en` is much more detailed and we should remove `mammoth_lvis_instruct4v_220k` to avoid redundancy and maximize the upweighting effect on the more detailed dataset.

| Dataset | Base Weight | Upweight | Changed | Packed Rows | Base Epochs | Upweight Epochs |
|---------|-----------|----------|---------|-------------|-------------|----------------|
| `openbee_honey_general_lvis_instructv4_v1_en` | 0.00560979 | 0.00560979 |  | 19,913 | 0.5770 | 0.5770 |
| `mammoth_lvis_instruct4v_220k` | 0.00040993 | 0.00040993 |  | 12,829 | 0.0654 | 0.0654 |

## VisualGenome (scene graphs, regions)

Combined base weight: 0.003335 (0.333% of total)

Notes:
- There is a repetition bug in the turns of the datasets where the first turn is repeated 3 times. This is the case for both `openbee_honey_grounding_and_counting_visualgenome_v1_en` and `mammoth_visualgenome_llava_next`. It is correct that we drop `mammoth_visualgenome_llava_next`.

| Dataset | Base Weight | Upweight | Changed | Packed Rows | Base Epochs | Upweight Epochs |
|---------|-----------|----------|---------|-------------|-------------|----------------|
| `openbee_honey_grounding_and_counting_visualgenome_v1_en` | 0.00243366 | 0.00627979 | **YES** | 8,574 | 0.5813 | 1.5000 |
| `mammoth_visualgenome_llava_next` | 0.00090122 | 0.00000000 | **YES** | 26,872 | 0.0687 | 0.0000 |

## TallyQA (counting)

Notes:
We should remove cambrian_10m_tallyqa to maximize the upweighting effect on `openbee_honey_grounding_and_counting_tallyqa_v1_en`, which is more detailed and of higher quality. We can also remove `the_cauldron_tallyqa` and `mammoth_tallyqa` since they are small and redundant with `openbee_honey_grounding_and_counting_tallyqa_v1_en`.

Combined base weight: 0.002946 (0.295% of total)

| Dataset | Base Weight | Upweight | Changed | Packed Rows | Base Epochs | Upweight Epochs |
|---------|-----------|----------|---------|-------------|-------------|----------------|
| `cambrian_10m_tallyqa` | 0.00148550 | 0.01676221 | **YES** | 22,886 | 0.1329 | 1.5000 |
| `openbee_honey_grounding_and_counting_tallyqa_v1_en` | 0.00110426 | 0.00299854 | **YES** | 4,094 | 0.5524 | 1.5000 |
| `the_cauldron_tallyqa` | 0.00033246 | 0.00000000 | **YES** | 8,269 | 0.0823 | 0.0000 |
| `mammoth_tallyqa` | 0.00002336 | 0.00000000 | **YES** | 764 | 0.0626 | 0.0000 |

## Localized Narratives

Combined base weight: 0.002050 (0.205% of total)

| Dataset | Base Weight | Upweight | Changed | Packed Rows | Base Epochs | Upweight Epochs |
|---------|-----------|----------|---------|-------------|-------------|----------------|
| `the_cauldron_localized_narratives` | 0.00204972 | 0.00204972 |  | 38,743 | 0.1084 | 0.1084 |

## IconQA

Notes:
- mammoth_iconqa and mammoth_internvl2_llama3_iconqa are very small and redundant with the openbee_honey datasets, so we should remove them to maximize the upweighting effect on the openbee_honey_iconqa datasets.
- the_cauldron_iconqa is lower quality than openbee_honey_iconqa and also small, so we should remove it as well to maximize the upweighting effect on the openbee_honey_iconqa datasets.
Combined base weight: 0.001974 (0.197% of total)

| Dataset | Base Weight | Upweight | Changed | Packed Rows | Base Epochs | Upweight Epochs |
|---------|-----------|----------|---------|-------------|-------------|----------------|
| `openbee_honey_general_iconqa_v1_en` | 0.00130883 | 0.00130883 |  | 4,625 | 0.5796 | 0.5796 |
| `openbee_honey_grounding_and_counting_iconqa_v1_en` | 0.00061595 | 0.00163989 | **YES** | 2,239 | 0.5634 | 1.5000 |
| `the_cauldron_iconqa` | 0.00004633 | 0.00004633 |  | 1,612 | 0.0589 | 0.0589 |
| `mammoth_iconqa` | 0.00000260 | 0.00000260 |  | 81 | 0.0656 | 0.0656 |
| `mammoth_internvl2_llama3_iconqa` | 0.00000066 | 0.00000066 |  | 21 | 0.0646 | 0.0646 |

## Other Grounding & Counting

Combined base weight: 0.001573 (0.157% of total)

| Dataset | Base Weight | Upweight | Changed | Packed Rows | Base Epochs | Upweight Epochs |
|---------|-----------|----------|---------|-------------|-------------|----------------|
| `openbee_honey_grounding_and_counting_tqa_v1_en` | 0.00144662 | 0.00376538 | **YES** | 5,141 | 0.5763 | 1.5000 |
| `openbee_honey_grounding_and_counting_mathv360k_vqa_as_v1_en` | 0.00011158 | 0.00029370 | **YES** | 401 | 0.5699 | 1.5000 |
| `openbee_honey_grounding_and_counting_other_v1_en` | 0.00001522 | 0.00004102 | **YES** | 56 | 0.5565 | 1.5000 |

## ShareGPT4V-SAM (grounded captions + SAM)

Notes:
- "openbee_honey_general_sharegpt4v_sam_v1_en" has floating point coordinates for bboxes [0.41, 0.19, 0.68, 0.45], and absolute coordinates, e.g. (335, 135, 505, 250)
Combined base weight: 0.001544 (0.154% of total).
- "mammoth_sharegpt4v_sam" is lower quality than "openbee_honey_general_sharegpt4v_sam_v1_en" and also small, so we should remove it to maximize the upweighting effect on "openbee_honey_general_sharegpt4v_sam_v1_en".

| Dataset | Base Weight | Upweight | Changed | Packed Rows | Base Epochs | Upweight Epochs |
|---------|-----------|----------|---------|-------------|-------------|----------------|
| `openbee_honey_general_sharegpt4v_sam_v1_en` | 0.00119330 | 0.00119330 |  | 4,281 | 0.5709 | 0.5709 |
| `mammoth_sharegpt4v_sam` | 0.00035072 | 0.00035072 |  | 10,508 | 0.0684 | 0.0684 |

## OCR BBox / Text Detection

Combined base weight: 0.000910 (0.091% of total)

| Dataset | Base Weight | Upweight | Changed | Packed Rows | Base Epochs | Upweight Epochs |
|---------|-----------|----------|---------|-------------|-------------|----------------|
| `ctw_bbox_v1` | 0.00030597 | 0.00030597 |  | 11,363 | 0.0551 | 0.0551 |
| `rects_cropped` | 0.00018266 | 0.00018266 |  | 5,360 | 0.0698 | 0.0698 |
| `hiertext_bbox_v1` | 0.00017337 | 0.00017337 |  | 5,503 | 0.0645 | 0.0645 |
| `rects_full` | 0.00014344 | 0.00014344 |  | 3,616 | 0.0812 | 0.0812 |
| `mammoth_rects_train` | 0.00010428 | 0.00010428 |  | 3,176 | 0.0672 | 0.0672 |

## Visual7W (pointing QA)

Combined base weight: 0.000846 (0.085% of total)

| Dataset | Base Weight | Upweight | Changed | Packed Rows | Base Epochs | Upweight Epochs |
|---------|-----------|----------|---------|-------------|-------------|----------------|
| `openbee_honey_general_visual7w_v1_en` | 0.00077008 | 0.00077008 |  | 2,691 | 0.5861 | 0.5861 |
| `mammoth_visual7w` | 0.00006009 | 0.00006009 |  | 1,889 | 0.0652 | 0.0652 |
| `the_cauldron_visual7w` | 0.00001630 | 0.00001630 |  | 1,073 | 0.0311 | 0.0311 |

## MovieNet

Combined base weight: 0.000168 (0.017% of total)

| Dataset | Base Weight | Upweight | Changed | Packed Rows | Base Epochs | Upweight Epochs |
|---------|-----------|----------|---------|-------------|-------------|----------------|
| `openbee_honey_grounding_and_counting_movienet_v1_en` | 0.00016794 | 0.00045483 | **YES** | 621 | 0.5538 | 1.5000 |

## Objects365 (object detection)

Combined base weight: 0.000081 (0.008% of total)

| Dataset | Base Weight | Upweight | Changed | Packed Rows | Base Epochs | Upweight Epochs |
|---------|-----------|----------|---------|-------------|-------------|----------------|
| `openbee_honey_general_objects365_v1_en` | 0.00008115 | 0.00032460 | **YES** | 296 | 0.5615 | 2.2459 |

---

## Duplicate / Overlapping Datasets

Datasets covering the same underlying data from different sources (openbee vs cambrian vs mammoth):

**clevr** (4 versions):

- `openbee_honey_grounding_and_counting_clevr_v1_en` — openbee, base_w=0.00624716, packed_rows=22844
- `cambrian_10m_clevr` — cambrian, base_w=0.00127667, packed_rows=29510
- `mammoth_clevr_700k` — mammoth, base_w=0.00082550, packed_rows=26434
- `the_cauldron_clevr` — other, base_w=0.00017814, packed_rows=6394

**clevr_math** (3 versions):

- `clevr_math_train` — other, base_w=0.00393962, packed_rows=22752
- `openbee_honey_grounding_and_counting_clevr_math_v1_en` — openbee, base_w=0.00006002, packed_rows=224
- `mammoth_clevr_math` — mammoth, base_w=0.00001563, packed_rows=507

**iconqa** (4 versions):

- `openbee_honey_general_iconqa_v1_en` — openbee, base_w=0.00130883, packed_rows=4625
- `openbee_honey_grounding_and_counting_iconqa_v1_en` — openbee, base_w=0.00061595, packed_rows=2239
- `the_cauldron_iconqa` — other, base_w=0.00004633, packed_rows=1612
- `mammoth_iconqa` — mammoth, base_w=0.00000260, packed_rows=81

**sharegpt4v_sam** (2 versions):

- `openbee_honey_general_sharegpt4v_sam_v1_en` — openbee, base_w=0.00119330, packed_rows=4281
- `mammoth_sharegpt4v_sam` — mammoth, base_w=0.00035072, packed_rows=10508

**super_clevr** (3 versions):

- `super_clevr_train` — other, base_w=0.00045754, packed_rows=2867
- `openbee_honey_grounding_and_counting_super_clevr_v1_en` — openbee, base_w=0.00016709, packed_rows=602
- `mammoth_super_clevr` — mammoth, base_w=0.00006136, packed_rows=1851

**tallyqa** (4 versions):

- `cambrian_10m_tallyqa` — cambrian, base_w=0.00148550, packed_rows=22886
- `openbee_honey_grounding_and_counting_tallyqa_v1_en` — openbee, base_w=0.00110426, packed_rows=4094
- `the_cauldron_tallyqa` — other, base_w=0.00033246, packed_rows=8269
- `mammoth_tallyqa` — mammoth, base_w=0.00002336, packed_rows=764

**visual7w** (3 versions):

- `openbee_honey_general_visual7w_v1_en` — openbee, base_w=0.00077008, packed_rows=2691
- `mammoth_visual7w` — mammoth, base_w=0.00006009, packed_rows=1889
- `the_cauldron_visual7w` — other, base_w=0.00001630, packed_rows=1073

**visualgenome** (2 versions):

- `openbee_honey_grounding_and_counting_visualgenome_v1_en` — openbee, base_w=0.00243366, packed_rows=8574
- `mammoth_visualgenome_llava_next` — mammoth, base_w=0.00090122, packed_rows=26872

