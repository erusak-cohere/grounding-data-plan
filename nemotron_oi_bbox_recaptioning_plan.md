# Nemotron VLM v2 OI BBox: Recaptioning Plan

## Source

`gs://cohere-data/vision/agent_trajectory/v4_filtered/uncompressed/nemotron_vlm_v2_oi_bbox_2_v1/`

## Dataset Overview

| Property | Value |
|----------|-------|
| Total samples | 1,664,533 |
| Total turns | 8,419,602 |
| Avg turns/sample | 5.1 |
| Shards | 1024 (`.jsonl`) |
| Images per sample | 1 |
| Bbox format | `[x1, y1, x2, y2]` integers in [0, 1000] |
| Source annotations | OpenImages V7 detection labels |

### Bbox quality assessment

**Range & geometry (25,440 bboxes from 2 shards):**
- All coordinates are integers in [0, 1000] — **no normalization needed**.
- **0 inverted boxes** (x2 < x1 or y2 < y1), **0 degenerate**, **0 zero-dimension**, **0 duplicates** within turns.
- Count consistency: **100%** — every time the response claims "there are N", there are exactly N bboxes.

**Size distribution:**
- Median area: ~16,600 (~1.7% of image). Mean: ~88,500 (right-skewed, normal for detection).
- 0.9% tiny (area < 100): sub-10px annotations of distant objects. Consider filtering area < 200.
- 1.1% near-full-image (area > 900k): objects that fill the frame. 2.1% of boxes have both dims > 800. Top categories: Building, Man, House — plausible but low-information for grounding.

**Semantic quality:**
- Categories are OpenImages-style fine-grained labels (Person, Man, Woman, Human face, Clothing, Footwear, Wheel, etc. — 600+ categories).
- Multi-bbox responses are common: ~52% of responses have 1 bbox, ~19% have 2, the rest have 3–153.
- Cross-turn bbox overlap is very low (0.45% of multi-turn samples), so different turns target genuinely different objects.
- OpenImages hierarchical labels cause frequent co-occurrence of e.g. "Man" + "Human face" + "Clothing" on the same image. Not a data error, but requires care during counting task generation (avoid conflating parent/child categories).

**Verdict: bboxes are high quality and can be reused as-is. Recommend light filtering of tiny boxes (area < 200) during final assembly.**

### Current format issues

1. **Special tokens**: Uses `<ref>Object</ref>:<box>[x1, y1, x2, y2]</box>` — must be converted to `<|box_start|>[x1, y1, x2, y2]<|box_end|>` and `<ref></ref>` must be removed.
2. **Grammar errors** (estimated ~20% of responses across the full dataset):
   - "there are 1 Man" → "there is 1 Man"
   - "The image contains 1 Flower. Here are their positions:" → "Here is its position:"
   - "The image has 1 Drum, and their positions are:" → singular form needed
   - Missing articles / awkward object-name pluralization ("where the Train are situated")
3. **Question monotony**: All questions ask for localization (text → bbox). There are ~25 template variants, but they all ask the same thing ("Locate every X", "Where are all the X?", etc.). No reverse-direction or counting tasks.
4. **No reasoning traces**: Responses jump straight to the answer with no `<START_THINKING>...<END_THINKING>` block.

---

## Recaptioning goals

Since the bboxes are already high quality and in the correct [0, 1000] format, we **skip steps 1 + 2** of the general recaptioning plan (coordinate harmonization and object identification / cross-referencing). We proceed directly to **Q+A generation (steps 3 + 4)**.

The recaptioning will produce three task types per image, reusing the existing bboxes and category labels as ground truth. Each assistant response must include a reasoning trace.

---

## Task types

### Task A — Grounding (text → bbox) [keep existing direction, fix quality]

The current task direction. The user asks "Where is the X?" and the model responds with bboxes.

**What changes:**
- Fix grammar (singular/plural agreement, articles).
- Replace `<ref>X</ref>:<box>[...]</box>` with `<|box_start|>[...]<|box_end|>`.
- Add a `<START_THINKING>...<END_THINKING>` reasoning block before the answer.
- Rephrase the question if the Qwen model finds it unnatural.

**Example (after recap):**

```
User: Where are all the flowers in this image?
Chatbot: <START_THINKING>I need to locate all instances of flowers in the image. I can see two flowers — one near the center-left area and another toward the upper-right corner.<END_THINKING>
There are 2 flowers in this image:
1. <|box_start|>[463, 488, 528, 631]<|box_end|>
2. <|box_start|>[818, 198, 852, 293]<|box_end|>
```

### Task B — Recognition (bbox → text) [new, reverse direction]

Provide the bboxes in the question and ask the model to identify what objects they contain. This complements Task A. The current category labels serve as ground truth for the Qwen model to validate against.

**Example (after recap):**

```
User: What objects are located at <|box_start|>[463, 488, 528, 631]<|box_end|> and <|box_start|>[818, 198, 852, 293]<|box_end|>?
Chatbot: <START_THINKING>I'll examine each highlighted region. The first region at [463, 488, 528, 631] shows a purple flower in bloom. The second region at [818, 198, 852, 293] contains a small yellow flower near the top of the image.<END_THINKING>
The objects are:
1. <|box_start|>[463, 488, 528, 631]<|box_end|> — A purple flower
2. <|box_start|>[818, 198, 852, 293]<|box_end|> — A small yellow flower
```

### Task C — Counting (derived from multi-bbox samples) [new]

For samples with multiple bboxes for the same category, convert into a counting task. Inspired by STEP3-VL-10B and Qwen3-VL, which explicitly construct counting data from detection annotations.

Three sub-formulations:

**C1 — Direct counting:**
```
User: How many cars are in this image?
Chatbot: <START_THINKING>I'll carefully scan the entire image for cars. I can see vehicles in the foreground and background. Let me count each one individually.<END_THINKING>
There are 3 cars in this image.
```

**C2 — Box-based counting (count + list):**
```
User: How many cars are in this image? Please provide their locations.
Chatbot: <START_THINKING>Let me identify each car and note its position. I see one large car in the foreground, and two smaller cars parked in the background.<END_THINKING>
There are 3 cars in this image:
1. <|box_start|>[50, 400, 600, 900]<|box_end|>
2. <|box_start|>[650, 300, 750, 450]<|box_end|>
3. <|box_start|>[800, 320, 880, 430]<|box_end|>
```

**C3 — Verification counting:**
```
User: Are there more than 5 people in this image?
Chatbot: <START_THINKING>Let me count all the people I can see. I count 1, 2, 3, 4, 5, 6, 7... yes, there are at least 7 people visible.<END_THINKING>
Yes, there are 7 people in this image, which is more than 5.
```

---

## Task distribution plan

Each `(image, category, bboxes[])` turn is randomly assigned **exactly one** task type to keep the dataset size manageable:

| Task | Weight | Eligibility | Notes |
|------|--------|-------------|-------|
| A — Grounding (text→bbox) | 40% | All turns | Fix grammar + add reasoning. |
| B — Recognition (bbox→text) | 30% | All turns | Reverse direction. |
| C — Counting | 30% | All turns (see fallback below) | Mix of C1/C2/C3 sub-types. |

**Single-bbox fallback:** When a turn is assigned Task C but has only 1 bbox:
- 80% → fall back to **A** (grounding)
- 20% → keep as **C** (counting with count=1, e.g. "Is there more than one X?" → "No, there is exactly 1.")

This ensures a small but non-trivial number of single-object counting examples exist in the dataset.

**Estimated final dataset size:** ~4.2M single-turn conversations (one per original turn), roughly the same volume as the source data but with diverse task types and reasoning traces.

---

## Processing pipeline

### Step 1 — Extract structured annotations from raw data (deterministic, no model)

Parse each sample to extract:
- `image` (base64 or reference)
- `image_width`, `image_height` (from metadata)
- For each turn: `category_label`, `bboxes` list (already [0, 1000])

This gives us a clean `(image, category, bboxes[])` triple per turn — the ground truth.

### Step 2 — Assign task type per turn (deterministic, no model)

For each `(image, category, bboxes[])` triple, randomly assign one task type:
- Draw from {A: 40%, B: 30%, C: 30%}.
- If C is drawn but `len(bboxes) < 2`: 80% fall back to A, 20% keep as C (count=1).
- If C is drawn, also randomly select a counting sub-type (C1, C2, or C3).
- Use a deterministic seed per `(sample_id, turn_index)` for reproducibility.

### Step 3 — Generate Q+A with Qwen (single pass)

For each turn, send one request to Qwen based on the assigned task type:

**Task A (Grounding):**
1. Pass image + category + bboxes to Qwen.
2. Qwen generates a natural grounding question, a reasoning trace, and the answer listing each bbox with `<|box_start|>...<|box_end|>` tokens.
3. Bboxes are **constrained** to match ground truth.

**Task B (Recognition):**
1. Construct a question embedding the bboxes with `<|box_start|>...<|box_end|>` tokens.
2. Pass image + question to Qwen.
3. Qwen generates a reasoning trace and a free-form description for each region.
4. Post-validation: check semantic consistency with the ground-truth category label.

**Task C (Counting):**
1. Pass image + category + count + bboxes to Qwen with the selected sub-type (C1/C2/C3).
2. Qwen generates a counting question, a reasoning trace, and the answer with the correct count (+ bbox list for C2).
3. Count and bboxes are **constrained** to match ground truth.

### Step 5 — Quality filtering

- Verify all `<|box_start|>...<|box_end|>` tokens are well-formed and contain valid [0, 1000] coordinates.
- Verify `<START_THINKING>...<END_THINKING>` is present in every assistant response.
- Remove degenerate outputs (empty reasoning, repetitive text, hallucinated bboxes not in ground truth).
- For Task B: check semantic alignment between Qwen's description and the category label.
- Grammar check: ensure no singular/plural mismatches in the generated text.

### Step 6 — Final assembly

Output format: `multimodal_ift` schema with:
```
User:  [image] <question, possibly containing <|box_start|>[x1, y1, x2, y2]<|box_end|>>
Chatbot: <START_THINKING>reasoning<END_THINKING> <answer, possibly containing <|box_start|>[x1, y1, x2, y2]<|box_end|>>
```

All coordinates are integers in [0, 1000], normalized relative to image dimensions.

---

## Qwen recaptioning prompt

See `nemotron_oi_bbox_recaptioning_prompt.md` for the full prompt to be used with the Qwen model.

---

## Tracking

| Step | Status |
|------|--------|
| Inspect data & assess bbox quality | DONE |
| Write recaptioning plan | DONE |
| Write Qwen recaptioning prompt | DONE |
| Step 1: Extract structured annotations | TODO |
| Step 2: Assign task types per turn | TODO |
| Step 3: Generate Q+A with Qwen (single pass) | TODO |
| Step 4: Quality filtering | TODO |
| Step 5: Final assembly | TODO |
