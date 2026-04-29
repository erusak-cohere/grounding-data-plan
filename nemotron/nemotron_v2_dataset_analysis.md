# Nemotron VLM Dataset v2: Full Analysis (Beyond oi_bbox)

**Date:** 2026-04-29
**Source:** [nvidia/Nemotron-VLM-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-VLM-Dataset-v2)
**License:** CC-BY-4.0 (wiki subsets: CC-BY-SA-4.0)

## 1. Executive Summary

Nemotron VLM v2 has 51 subsets totaling ~8.1M samples. We have already investigated `oi_bbox_{1,2,3}` (~4.5M samples, high-quality bboxes from OpenImages V7). This document covers the **remaining 48 subsets** and identifies what they can add on top of the openbee data mix.

**Key finding:** Most of the non-oi_bbox subsets overlap with datasets already in the openbee mix. The main unique value-adds from nemotron beyond oi_bbox are:
1. **Multilingual Wikipedia OCR** (2M samples, 10 languages) -- unique scale for multilingual OCR
2. **Synthetic sparse table parsing** (100k) -- complements existing table datasets
3. **CoT-annotated versions of existing datasets** -- step-by-step reasoning for chartqa, docvqa, etc.
4. **Video understanding** (not downloaded, ~860k samples) -- entirely new modality vs openbee

---

## 2. What's Downloaded on GCS

All non-video image subsets are at:
`gs://cohere-data/vision/agent_trajectory/v4_filtered/uncompressed/nemotron_vlm_v2_*`

### 2.1 Downloaded subsets (35 subsets)

| Subset | Samples | Type | Labels | Notes |
|--------|---------|------|--------|-------|
| oi_bbox_1 | 1,664,533 | Grounding (bbox QA) | Public (OI V7) | **Already investigated** |
| oi_bbox_2 | 1,664,533 | Grounding (bbox QA) | Public (OI V7) | **Already investigated** |
| oi_bbox_3 | 1,128,326 | Grounding (bbox QA) | Public (OI V7) | **Already investigated** |
| wiki_de | 200,000 | OCR (image→markdown) | Public | Wikipedia page screenshots |
| wiki_en | 200,000 | OCR (image→markdown) | Public | Wikipedia page screenshots |
| wiki_es | 200,000 | OCR (image→markdown) | Public | Wikipedia page screenshots |
| wiki_fr | 200,000 | OCR (image→markdown) | Public | Wikipedia page screenshots |
| wiki_it | 200,000 | OCR (image→markdown) | Public | Wikipedia page screenshots |
| wiki_ja | 200,000 | OCR (image→markdown) | Public | Wikipedia page screenshots |
| wiki_ko | 200,000 | OCR (image→markdown) | Public | Wikipedia page screenshots |
| wiki_nl | 200,000 | OCR (image→markdown) | Public | Wikipedia page screenshots |
| wiki_pt | 200,000 | OCR (image→markdown) | Public | Wikipedia page screenshots |
| wiki_zh | 200,000 | OCR (image→markdown) | Public | Wikipedia page screenshots |
| chartqa_cot | 45,710 | Chart QA + reasoning | Qwen-labels | Multi-turn: CSV conversion + QA |
| docvqa_cot | 36,333 | Document QA + reasoning | Qwen-labels | Short-answer extraction from docs |
| infographicsvqa_cot | 19,548 | Infographic QA + reasoning | Qwen-labels | Short-answer from infographics |
| fintabnet_cot | 8,356 | Financial table QA + reasoning | Qwen-labels | Financial document parsing |
| tabmwp_cot | 20,305 | Table math word problems | Qwen-labels | Math over table images |
| geomverse_cot | 9,298 | Geometry reasoning | GLM-labels | Long step-by-step math |
| unigeo_cot | 9,728 | Geometry (Chinese) | GLM-labels | Chinese geometry MCQ |
| mapqa_cot | 16,832 | Map QA + reasoning | GLM-labels | Choropleth map QA |
| plotqa_cot | 16,256 | Plot QA + reasoning | GLM-labels | Scientific plot QA |
| mulberry_cot_1 | 191,332 | Multi-task reasoning | GLM-labels | Diverse visual reasoning MCQ |
| mulberry_cot_2 | 103,763 | Multi-task reasoning | Qwen-labels | Diverse visual reasoning MCQ |
| llava_cot_100k | 63,013 | General visual QA | GLM-labels | MCQ across diverse images |
| mantis_instruct_cot | 67,723 | Multi-image instruction | GLM-labels | 2-image comparison/reasoning |
| visual7w_telling_cot | 62,592 | Visual QA (7W) | GLM-labels | "Telling" subset of Visual7W |
| visual_web_instruct_cot | 47,829 | Web-based visual instruction | GLM-labels | Math/reasoning from web images |
| nights_cot | 12,906 | Image similarity | GLM-labels | 3-image similarity comparison |
| sparsetables | 100,000 | Table → HTML parsing | Synthetic | Synthetic sparse table images |
| ccpdf_multipage_1 | 6,100 | Multi-page PDF QA | Qwen-labels | Document understanding |
| ccpdf_nv_notables | 14,234 | PDF OCR (no tables) | Human-labels | High-quality document reading |
| ccpdf_nv_qa | 1,668 | PDF QA | Qwen-labels | QA over PDF pages |
| ccpdf_nv_tables | 4,249 | PDF table extraction | Human-labels | Table-specific OCR |
| hiertext | 514 | Hierarchical text OCR | Qwen-labels | Very small, possibly not useful |

### 2.2 NOT downloaded -- Video subsets (16 subsets, ~860k samples)

| Subset | Samples | Source Video Dataset |
|--------|---------|---------------------|
| kinetics_k710 | ~431k | Kinetics-710 action recognition |
| nextqa | 34,132 | NeXT-QA video reasoning |
| clevrer | 40,000 | CLEVRER physics reasoning |
| ego_exo_learn | 36,373 | Ego-exo action understanding |
| hacs | 31,223 | Human Action Clips |
| activity_net_1 | 10,021 | ActivityNet temporal reasoning |
| activity_net_2 | 9,064 | ActivityNet temporal reasoning |
| perception_test_1 | 7,392 | Perception Test |
| perception_test_2 | 2,135 | Perception Test |
| perception_test_cot | 4,977 | Perception Test + reasoning |
| hirest_1 | 822 | HiREST video summarization |
| hirest_2 | 525 | HiREST video summarization |
| youcook2_1 | 1,180 | YouCook2 cooking videos |
| youcook2_2 | 2,270 | YouCook2 cooking videos |
| breakfast_actions | 1,204 | Breakfast Actions |
| ccpdf_multipage_2 | 420 | Multi-page PDF (possibly video-like) |

---

## 3. Data Format Observations

All downloaded subsets share the same standardized format:
- Shards named `trajectories-shard-XXXXX-of-YYYYY.jsonl`
- Each line: `{metadata, id, source, split, version, preamble, turns}`
- Turns contain `{role: "User"|"Chatbot", contents: [{content_type: "text"|"image", ...}]}`
- Images embedded as base64 JPEG in `image_url` field
- Each turn has `qwen3_0_6b_emb` text embeddings in metadata

### 3.1 Reasoning traces

Despite the "cot" suffix, **most samples do NOT contain explicit thinking traces** (`<think>`, `<START_THINKING>`, etc.) in our standardized format. Instead:
- The `reasoning_effort` flag in preamble is set to `"ON"` for CoT subsets
- Some datasets (geomverse_cot, visual_web_instruct_cot) have long step-by-step answers that constitute implicit chain-of-thought
- Other "cot" datasets (mulberry_cot, llava_cot_100k) have very short answers ("Final Answer: X") -- the original CoT from NVIDIA was likely stripped during standardization
- The original NVIDIA data used `<think>...</think>` tags for GLM-labeled datasets and structured reasoning for Qwen-labeled ones

### 3.2 Multi-turn vs single-turn

- Most CoT datasets: single user+chatbot turn pair
- chartqa_cot: multi-turn (image → CSV conversion → QA question → short answer)
- mantis_instruct_cot: single turn but **multi-image** (2 images per sample)
- nights_cot: single turn with **3 images** (reference + 2 candidates)

---

## 4. Cross-Reference with Openbee/Current Data Mix

### 4.1 High overlap with openbee (nemotron offers CoT variant of same base data)

| Nemotron Subset | Openbee Equivalent | Nemotron Adds |
|----------------|-------------------|---------------|
| chartqa_cot (45.7k) | openbee_honey_chart_chartqa | Multi-turn: CSV + QA with reasoning flag |
| docvqa_cot (36.3k) | openbee_honey_document_docvqa | Short-answer QA with reasoning flag |
| infographicsvqa_cot (19.5k) | openbee_honey_document_infographicvqa | QA with reasoning flag |
| fintabnet_cot (8.4k) | openbee_honey_document_fintabnet | Financial QA with reasoning flag |
| plotqa_cot (16.3k) | openbee_honey_chart_plotqa | Plot QA with reasoning flag |
| tabmwp_cot (20.3k) | openbee_honey_chart_tabmwp | Math word problems with reasoning flag |
| mapqa_cot (16.8k) | openbee_honey_stem_mapqa | Map QA with reasoning flag |
| geomverse_cot (9.3k) | openbee_honey_stem_geomverse | Step-by-step geometry solutions |
| visual7w_telling_cot (62.5k) | openbee_honey_general_visual7w | 7W QA with reasoning flag |
| mulberry_cot_{1,2} (295k) | openbee_honey_general_cauldron_mulberry | Multi-task MCQ with reasoning flag |
| ccpdf_nv_* (20.2k) | openbee_honey_document_* (various) | PDF-specific OCR and QA |

**Verdict:** These are the same underlying source datasets with CoT annotations added by NVIDIA using Qwen/GLM. The openbee versions are already in our mix. The nemotron versions could be useful if we want step-by-step reasoning variants, but currently the CoT traces appear stripped in our standardized format, making them near-duplicates of what we already have.

### 4.2 Moderate overlap (nemotron has different task formulation)

| Nemotron Subset | Similar in Current Mix | Difference |
|----------------|----------------------|------------|
| llava_cot_100k (63k) | mammoth_llava_instruct_300k, openbee_honey_general_llava_instruct_300k | MCQ format with reasoning flag vs open-ended |
| mantis_instruct_cot (67.7k) | No direct equivalent | **Multi-image** reasoning (2 images) |
| nights_cot (12.9k) | openbee_honey_general_dreamsim, contrastivecaption | **3-image** similarity comparison |
| visual_web_instruct_cot (47.8k) | No direct equivalent | Math/reasoning from web screenshots, multilingual |

### 4.3 Low overlap -- Unique value from nemotron

| Nemotron Subset | What It Provides | Why It's Unique vs Openbee |
|----------------|-----------------|--------------------------|
| **wiki_{10 langs}** (2M total) | Wikipedia page → Markdown/LaTeX OCR | **Scale:** 200k per language for de/en/es/fr/it/ja/ko/nl/pt/zh. Openbee has OCR (arxiv_ocr, ctw, iam, etc.) but nothing at this scale for multilingual structured document OCR. The task is full-page rendering including tables as LaTeX/HTML. |
| **sparsetables** (100k) | Synthetic table image → HTML | **Synthetic diversity:** Random table structures with sparse cells, merged cells, varied layouts. Openbee has real table datasets (fintabnet, scitsr, robut_*) but this synthetic data covers edge cases those miss. |
| **oi_bbox_{1,2,3}** (4.5M) | Object grounding with bboxes | **Already known.** Openbee has objects365 and visualgenome for grounding, but nemotron's OI bbox data is much larger (4.5M vs ~200k) with 600+ categories and high-quality coordinates. |
| **unigeo_cot** (9.7k) | Chinese geometry MCQ | Openbee has geoqa in Chinese, but unigeo is a different dataset. Small but fills a niche. |

### 4.4 Not applicable (video -- entirely different modality)

The 16 video subsets (~860k samples) are a completely different modality. Openbee is image-only, so these don't overlap at all. If/when we support video SFT, these would be the primary interest:
- **kinetics_k710** (~431k) -- massive action recognition dataset
- **clevrer** (40k) -- physics reasoning over synthetic videos
- **nextqa** (34k) -- temporal and causal video QA
- **ego_exo_learn** (36k) -- egocentric/exocentric view understanding

---

## 5. Recommendations

### 5.1 Worth adding to the data mix

1. **Wiki multilingual OCR (2M samples)**
   - 10 languages at 200k each
   - Task: full-page Wikipedia screenshot → structured Markdown with LaTeX tables
   - Would significantly boost multilingual OCR capabilities
   - Already downloaded and standardized
   - **Caveat:** CC-BY-SA-4.0 license (not CC-BY-4.0)

2. **sparsetables (100k)**
   - Synthetic table images → HTML extraction
   - Complements real-world table datasets in openbee (fintabnet, scitsr, etc.)
   - Already downloaded and standardized

3. **mantis_instruct_cot (67.7k)**
   - Multi-image reasoning (2 images per sample)
   - Unique format: "Is this statement about these 2 images true/false?"
   - Currently openbee has limited multi-image samples

### 5.2 Potentially worth adding (if CoT traces can be recovered)

4. **CoT variants of existing datasets** (chartqa_cot, docvqa_cot, geomverse_cot, visual_web_instruct_cot)
   - The original NVIDIA data has `<think>` traces that add step-by-step reasoning
   - In our standardized format, these traces were mostly stripped
   - If we can re-standardize preserving the thinking traces, these become valuable as reasoning training data
   - Without the traces, they're near-duplicates of what openbee already provides

### 5.3 Probably skip

5. **mulberry_cot_{1,2}, llava_cot_100k, visual7w_telling_cot** -- These are short-answer MCQ versions of data already well-covered by openbee
6. **hiertext (514 samples)** -- Too small to matter
7. **ccpdf_nv_* (20.2k total)** -- Overlaps with openbee document datasets
8. **nights_cot (12.9k)** -- Niche image similarity task, small
9. **unigeo_cot (9.7k)** -- Chinese geometry, small and niche

### 5.4 Future consideration (video)

10. Video subsets (~860k) are not downloaded. If/when video SFT is supported, kinetics_k710, nextqa, clevrer, and ego_exo_learn would be the most interesting.

---

## 6. Sample Data Examples

### Wiki OCR (wiki_en)
```
User: [IMAGE] Convert the text in this image into a readable document in mathpix markdown format
Chatbot: ## History
Planning for a chapel in Værlandet began in the mid-1930s when some fundraising
was carried out and a new cemetery was built...
```

### ChartQA CoT (chartqa_cot) -- Multi-turn
```
Turn 0 User: [IMAGE] Convert the essential information from the graph into a CSV format.
Turn 1 Chatbot: Year,Number of Production Sites\n2009,115\n2010,110...
Turn 2 User: How many ice cream production locations were there in 2014?
Turn 3 Chatbot: 100
```

### Sparse Tables (sparsetables)
```
User: [IMAGE] Parse the tabular information from this image and provide it as HTML.
Chatbot: <table><tr><td></td><td><b>XTY</b></td><td></td>...
```

### Geomverse CoT (geomverse_cot) -- Long step-by-step
```
User: [IMAGE] If the area of the ABCD rectangle is 114, the ABEFG shape is a
combination of a rectangle and an equilateral triangle... compute the length of AD.
Chatbot: The area of the BEHIJ shape is 114 and the length of the EH side is 6,
so OtherSide * 6 + sqrt(3)/4 * 6^2 = 114, so OtherSide * 6 = 114 - 15.48 = 98.52.
Therefore, the length of the BE side is 98.52/6 = 16.42...
```

### Mantis Instruct CoT (mantis_instruct_cot) -- Multi-image
```
User: Here is a statement describing 2 images: One dog is sleeping with a stuffed
animal. Is it true or false?
[IMAGE 1] [IMAGE 2]
Options: (A) False (B) True
Chatbot: A
```

### Visual Web Instruct CoT (visual_web_instruct_cot) -- Step-by-step math
```
User: [IMAGE] Which of the following statements is true for this formula?
A. When b > 2, a is negative. B. When b < 2, a is positive...
Chatbot: To determine the correct statement, analyze the formula a = 1/2 b + 1...
Option C: When b < -2, substitute b = -3, giving a = -0.5 (negative). This holds...
Answer: C
```

---

## 7. GCS Paths

All downloaded at:
```
gs://cohere-data/vision/agent_trajectory/v4_filtered/uncompressed/nemotron_vlm_v2_{subset_name}_v1/
```

Shard format: `trajectories-shard-XXXXX-of-YYYYY.jsonl`
