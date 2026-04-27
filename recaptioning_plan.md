# Grounding Data: Coordinate Harmonization & Recaptioning Plan

## 1. Coordinate Harmonization

All bbox/point coordinates must be converted to **integer [0, 1000]** format and wrapped with special tokens: `<|box_start|>[x1, y1, x2, y2]<|box_end|>`.

### 1.1 Existing datasets needing reprocessing

These datasets already exist in the training mix but use inconsistent coordinate formats:

- [ ] `openbee_honey_grounding_and_counting_visualgenome_v1_en` — [0-1] floats, e.g. `[0.708,0.787,0.728,0.838]`
- [ ] `hiertext_bbox_v1` — [0-1] floats, e.g. `"bbox": [0.07, 0.22, 0.61, 0.27]`
- [ ] `ctw_bbox_v1` — [0-1] floats, e.g. `[0.55, 0.38, 0.56, 0.39]`
- [ ] `openbee_honey_general_sharegpt4v_sam_v1_en` — **mixed formats**: [0-1] floats for input regions AND raw pixel integers for output bboxes
- [ ] `pixmo_point_expl` — percentage points in `<points points="(21.5, 10.6), ...">` format

### 1.2 New datasets (ingest directly with [0, 1000] + special tokens)

- [x] RefCOCO / RefCOCO+ / RefCOCOg — ingested with [0, 1000]; **needs special token wrapping + recap** (see §2)
- [ ] Objects365 (200K subset) — ingest with [0, 1000] + special tokens + recap
- [ ] OpenImages V7 Detection (100K subset) — ingest with [0, 1000] + special tokens + recap
- [ ] Long Grounded Thoughts (stage1) — assess coordinate format, harmonize if needed
- [ ] VisionFoundry-10K — assess coordinate format, harmonize if needed
- [ ] FSC147 — counting dataset, assess if bbox coords are present
- [ ] OODVQA — counting/VQA, likely no explicit coords

---

## 2. Recaptioning Pipeline

The goal is to transform raw grounding annotations (bbox + short label/expression) into high-quality, diverse VQA conversations. This is done per-dataset after ingestion.

### 2.1 RefCOCO Family (RefCOCO + RefCOCO+ + RefCOCOg)

**Preprocessing:**

- [ ] **Deduplicate by (image_id, bbox)** across all three datasets. For each unique (image_id, bbox) pair, collect all referring expressions from all three sources. This yields ~28K unique images with their associated bbox regions, each annotated with 1–N expressions of varying style (short/location-aware from RefCOCO, appearance-only from RefCOCO+, long from RefCOCOg).

**Recap steps:**

- [ ] **Step 1 — Object identification.** For each unique (image_id, bbox), pass the image with the bbox highlighted (or cropped) to the recap model. Ask: *"What object is in this region?"* This produces a model-generated object label for the region.

- [ ] **Step 2 — Cross-reference with existing expressions.** Compare the model's object identification against the pooled referring expressions from RefCOCO/+/g for that (image_id, bbox). Measure semantic alignment (e.g., does the model say "dog" when expressions say "brown dog on the left"?). Use cases:
  - **High agreement** → the bbox and its content are reliable; proceed to question generation.
  - **Low agreement** → flag for review or discard. May indicate a bad bbox, ambiguous region, or model hallucination.
  - **Calibration signal** — the alignment rate across the dataset can inform whether the recap model needs few-shot examples or system prompt tuning before bulk generation.

- [ ] **Step 3 — Question generation.** Using the object label (from step 1) and the bbox, generate diverse questions. Template categories:
  - **Localization (text → bbox):** *"Where is the [object description]?"* → answer contains the bbox.
  - **Recognition (bbox → text):** *"What is the object at `<|box_start|>[x1, y1, x2, y2]<|box_end|>`?"* → answer describes the object.
  - **Descriptive:** *"Describe the object in the highlighted region."* → richer free-form answer.
  - **Relational:** *"What is to the left of the [object]?"* → spatial reasoning (if multiple objects in the image).
  - Use multiple phrasings per template to avoid pattern memorization.

- [ ] **Step 4 — Answer generation.** For each (image, question) pair, generate the answer with the recap model. For localization questions, the answer should include the bbox wrapped in special tokens. For recognition/descriptive questions, the answer is free-form text.

- [ ] **Step 5 — Quality filtering.** Validate generated QA pairs:
  - Check answer consistency: if a localization answer is expected, verify it contains a valid bbox.
  - Filter out low-confidence or degenerate outputs (empty, repetitive, or nonsensical).
  - Optionally back-verify: for recognition answers, confirm the description matches the region.

- [ ] **Step 6 — Final dataset assembly.** Keep the VQA outputs (image + question + answer). Drop the original raw referring expressions. The final format uses `<|box_start|>[x1, y1, x2, y2]<|box_end|>` wherever a bbox appears in the text (question or answer side).

### 2.2 Objects365

- [ ] Download 200K category-balanced subset
- [ ] Recap: same pipeline as §2.1, starting from (image, bbox, category_label) triples. Step 2 cross-references against the category label instead of referring expressions.

### 2.3 OpenImages V7 Detection

- [ ] Download 100K category-balanced subset
- [ ] Recap: same pipeline as §2.1, starting from (image, bbox, class_label, confidence) tuples.

### 2.4 Other datasets

Datasets like VisualGenome, ShareGPT4V-SAM, hiertext, ctw already have richer annotations (scene graphs, grounded captions, OCR text). Their recap pipelines will be tailored per-dataset but follow the same principles: harmonize coordinates → wrap with special tokens → generate diverse VQA → quality filter.

---

## 3. Output Format

All recaptioned data uses the `multimodal_ift` schema:

```
User:  [image] <question, possibly containing <|box_start|>[x1, y1, x2, y2]<|box_end|>>
Chatbot: <answer, possibly containing <|box_start|>[x1, y1, x2, y2]<|box_end|>>
```

Coordinates are always integers in [0, 1000], normalized relative to image dimensions.

---

## 4. Tracking

| Dataset | Ingested | Coords [0,1000] | Special tokens | Recaptioned |
|---------|----------|-----------------|----------------|-------------|
| RefCOCO | Yes | Yes | TODO | TODO |
| RefCOCO+ | Yes | Yes | TODO | TODO |
| RefCOCOg | Yes | Yes | TODO | TODO |
| Objects365 (200K) | TODO | — | — | TODO |
| OpenImages V7 (100K) | TODO | — | — | TODO |
| LGT stage1 | TODO | — | — | TODO |
| VisionFoundry-10K | TODO | — | — | TODO |
| FSC147 | TODO | — | — | — |
| OODVQA | TODO | — | — | — |
| VisualGenome (existing) | Yes | TODO | TODO | TODO |
| hiertext_bbox (existing) | Yes | TODO | TODO | — |
| ctw_bbox (existing) | Yes | TODO | TODO | — |
| ShareGPT4V-SAM (existing) | Yes | TODO | TODO | TODO |
| PixMo points (existing) | Yes | TODO | TODO | — |
