# Nemotron VLM v2 OI BBox: Recaptioning Plan & Pipeline

## 1. Dataset Overview

**Source:** `gs://cohere-data/vision/agent_trajectory/v4_filtered/uncompressed/nemotron_vlm_v2_oi_bbox_2_v1/`

![Example image from Nemotron VLM v2 OI BBox dataset](./nemotron_example.png)

| Property | Value |
|----------|-------|
| Total samples (images) | 1,664,533 |
| Total turns (User + Chatbot) | 8,419,602 |
| Avg turns/sample | 5.1 (i.e. ~2.5 Q+A pairs per image) |
| Shards | 1024 (`.jsonl`) |
| Images per sample | 1 (base64 JPEG embedded in turn 0) |
| Bbox format | `[x1, y1, x2, y2]` integers in [0, 1000] |
| Source annotations | OpenImages V7 detection labels (600+ categories) |

### 1.1 Bbox quality assessment

**Geometry (25,440 bboxes sampled from 2 shards):**
- All coordinates are integers in [0, 1000] -- no normalization needed.
- 0 inverted boxes (x2 < x1 or y2 < y1), 0 degenerate, 0 zero-dimension, 0 duplicates within turns.
- Count consistency: 100% -- every time the response claims "there are N", there are exactly N bboxes.

**Size distribution:**
- Median area: ~16,600 (~1.7% of image). Mean: ~88,500 (right-skewed, normal for detection).
- 0.9% tiny (area < 100): sub-10px annotations of distant objects. Consider filtering area < 200.
- 1.1% near-full-image (area > 900k): objects that fill the frame. Top categories: Building, Man, House.

**Semantic quality:**
- Multi-bbox responses are common: ~52% have 1 bbox, ~19% have 2, the rest have 3-153.
- Cross-turn bbox overlap is very low (0.45%), confirming different turns target genuinely different objects.
- OpenImages hierarchical labels cause frequent co-occurrence of e.g. "Man" + "Human face" + "Clothing" on the same image. Requires care during counting task generation.

**Verdict: bboxes are high quality and can be reused as-is.**

### 1.2 Multi-turn structure analysis

63% of samples are multi-turn (2+ Q+A pairs on the same image). Key findings:
- **100% of multi-turn samples have distinct categories** across turns -- every turn asks about a different object type.
- **99.7% have completely disjoint bboxes** across turns.
- Each multi-turn sample is essentially a **complete scene annotation** (e.g., Houses x4 + Trees x2 + Doors x8 on one image).
- **45% of multi-turn samples** have at least 2 categories with exactly 1 spatially-separated bbox each -- suitable for relational questions.

### 1.3 Current format issues (why we need recaptioning)

1. **Special tokens**: Uses `<ref>Object</ref>:<box>[x1, y1, x2, y2]</box>` -- must be converted to `<|box_start|>[x1, y1, x2, y2]<|box_end|>` and `<ref></ref>` must be removed.
2. **Grammar errors** (~20% of responses): "there are 1 Man", "Here are their positions" for singular items, awkward category-name pluralization.
3. **Question monotony**: All ~25 question templates ask the same thing (text -> bbox localization). No reverse-direction, counting, or relational tasks.
4. **No reasoning traces**: Responses jump straight to the answer with no `<START_THINKING>...<END_THINKING>` block.
5. **Multi-turn structure wastes potential**: Each turn is independent; the rich cross-category scene information is never exploited.

---

## 2. Consolidation & Recaptioning Goals

### 2.1 Consolidation

We **consolidate all turns of each sample into a single scene annotation**:

```
Before (multi-turn):
  Turn 0: User asks about House    -> Chatbot: 4 house bboxes
  Turn 1: User asks about Tree     -> Chatbot: 2 tree bboxes
  Turn 2: User asks about Window   -> Chatbot: 8 window bboxes

After (consolidated):
  Scene annotation for this image:
    House:  [[61,55,606,521], [569,147,811,460], [834,274,931,405], [933,211,999,401]]
    Tree:   [[291,25,382,97], [954,60,999,250]]
    Window: [[5,198,18,257], [138,746,143,805], ...]
```

This consolidated scene annotation becomes the ground truth passed to Qwen. It enables richer questions that reference multiple object types, spatial relationships, and full-scene understanding.

### 2.2 Recaptioning goals

- One Q+A conversation per image (1 sample = 1 output).
- Each sample is randomly assigned **exactly one** of four task types (A/B/C/D).
- Every assistant response includes a `<START_THINKING>...<END_THINKING>` reasoning trace.
- All bboxes use `<|box_start|>[x1, y1, x2, y2]<|box_end|>` tokens.

**Estimated final dataset size:** ~1.66M single-turn conversations (one per image).

---

## 3. Task Types

### Task A -- Grounding (text -> bbox)

The user asks "Where is the X?" and the model responds with bboxes. One category is selected from the consolidated scene annotation.

**Example:**

```
User: Where are all the flowers in this image?
Chatbot: <START_THINKING>I need to locate all instances of flowers in the image. I can
see two flowers -- one near the center-left area and another toward the upper-right
corner.<END_THINKING>
I found 2 flowers in this image:
1. <|box_start|>[463, 488, 528, 631]<|box_end|>
2. <|box_start|>[818, 198, 852, 293]<|box_end|>
```

### Task B -- Recognition (bbox -> text)

Provide bboxes in the question and ask the model to identify what they contain. Bboxes can come from one or more categories in the consolidated annotation. The category labels serve as ground truth for validation.

**Example (cross-category, leveraging consolidation):**

```
User: What can you see at <|box_start|>[61, 55, 606, 521]<|box_end|>,
<|box_start|>[291, 25, 382, 97]<|box_end|>, and
<|box_start|>[5, 198, 18, 257]<|box_end|>?
Chatbot: <START_THINKING>I'll examine each region. The first is a large area in the
upper-left showing a white two-story house. The second is a small area at the top
containing a tree canopy. The third is a narrow vertical strip on the left edge
showing a window frame.<END_THINKING>
The objects at the specified locations are:
1. <|box_start|>[61, 55, 606, 521]<|box_end|> -- A white two-story house with a red roof.
2. <|box_start|>[291, 25, 382, 97]<|box_end|> -- A tree canopy with green leaves.
3. <|box_start|>[5, 198, 18, 257]<|box_end|> -- A narrow window with a dark frame.
```

### Task C -- Counting (derived from detection annotations)

Inspired by [STEP3-VL-10B](https://arxiv.org/abs/2601.09668) and [Qwen3-VL](https://arxiv.org/abs/2511.21631). One category is selected from the consolidated annotation.

Three sub-formulations:

**C1 -- Direct counting:** "How many X?" -> count only, no bboxes.

**C2 -- Box-based counting:** "How many X? Show their locations." -> count + bbox list.

**C3 -- Verification counting:** "Are there more/fewer/exactly N X?" -> yes/no + actual count.
- "More/fewer than N": pick N = count +/- random(1,3).
- "Exactly N": pick N = count (yes) or N = count +/- random(1,3) (no).
- Mix these roughly equally.

### Task D -- Relational (spatial reasoning between objects)

For images where 2+ categories each have exactly 1 bbox and the bboxes are spatially separated, ask about spatial relationships between them.

**Eligibility filters:**
- At least 2 categories in the consolidated annotation with exactly 1 bbox each.
- The selected pair must have IoU < 0.1 (genuinely separate objects).
- Neither bbox covers >70% of the image area (no full-image boxes).
- Categories should not be hierarchically related (e.g., skip "Man" + "Human face" or "Man" + "Clothing" where one is a part of the other).

**Spatial relations** (derived deterministically from bbox coordinates):
- Left/right: compare x-center of the two bboxes.
- Above/below: compare y-center.
- Larger/smaller: compare bbox area.
- Near/far: compare Euclidean distance between centers relative to image diagonal.

**Example:**

```
User: What is the spatial relationship between the tree and the car in this image?
Chatbot: <START_THINKING>The tree is located at <|box_start|>[3, 27, 436, 478]<|box_end|>, which places it
in the upper-left portion of the image. The car is at <|box_start|>[11, 385, 999, 744]<|box_end|>, which
occupies the lower-right area. Comparing their positions, the tree is above and to
the left of the car.<END_THINKING>
The tree <|box_start|>[3, 27, 436, 478]<|box_end|> is above and to the left of
the car <|box_start|>[11, 385, 999, 744]<|box_end|>.
```

**Question format variants:**
- "Is the X to the right of the Y?" (yes/no)
- "What is the spatial relationship between the X and the Y?" (open-ended)
- "Which is larger, the X or the Y?" (comparison)

---

## 4. Task Distribution

Each **image** (consolidated sample) is randomly assigned exactly one task type:

| Task | Weight | Eligibility | Notes |
|------|--------|-------------|-------|
| A -- Grounding (text->bbox) | 30% | All samples | Pick 1 category from scene. |
| B -- Recognition (bbox->text) | 25% | All samples | Pick bboxes from 1+ categories. |
| C -- Counting | 25% | All samples (see fallback) | Pick 1 category; mix of C1/C2/C3. |
| D -- Relational | 20% | Multi-turn with eligible pairs (~45% of multi-turn) | Spatial reasoning. |

**Fallbacks:**
- Task C with only 1 bbox: 80% fall back to A or B (50/50), 20% keep as C (count=1).
- Task D on ineligible samples (single-turn or no qualifying pairs): fall back to A or B (50/50).

**Estimated final dataset size:** ~1.66M single-turn conversations (one per image).

---

## 5. Pipeline Architecture

```
JSONL Shard --> Parser --> Consolidator --> Task Assigner --> Prompt Builder --> Qwen VLM API --> Validator --> Output JSONL
```

### 5.1 Flow

1. **Extract**: Read a JSONL shard, parse each sample's raw turns.
2. **Consolidate**: Merge all turns of each sample into a single scene annotation: `(image_b64, image_meta, [(category_1, bboxes_1), (category_2, bboxes_2), ...])`.
3. **Assign**: Deterministically assign a task type (A/B/C1/C2/C3/D) per sample, seeded by `sample_id` for reproducibility. For tasks A/B/C, also select which category or bboxes from the scene to use.
4. **Build prompt**: Call the task-specific prompt function. Each function returns `(system_prompt, user_message)` -- the Qwen model only sees the prompt for its assigned task. The full scene annotation is provided as context.
5. **Inference**: Send image + messages to the Qwen VLM API (OpenAI-compatible chat completions).
6. **Validate**: Check the model output (bbox tokens, thinking tags, count/relation consistency). Retry once on failure, then discard.
7. **Write**: Emit a single-turn conversation to the output JSONL.

### 5.2 File structure

```
nemotron_recaptioning/
  config.py          # Tunables: API endpoint, model name, task weights, concurrency, paths
  parser.py          # Extract raw turns from JSONL
  consolidator.py    # Merge multi-turn into scene annotations
  task_assigner.py   # Deterministic random task assignment + category/bbox selection
  prompts.py         # 6 prompt-builder functions (A, B, C1, C2, C3, D)
  api_client.py      # Async wrapper for Qwen VLM API with retry logic
  validator.py       # Per-task output validation
  pipeline.py        # Main orchestrator: shard iteration, async concurrency, CLI
```

### 5.3 Key module: `prompts.py`

Six functions, one per task variant. Each receives the image and the relevant slice of the consolidated scene annotation. The model never sees prompts for other task types.

```python
def build_prompt_grounding(image_b64, category, bboxes, scene_annotation, image_meta):
    """Task A: text -> bbox. Returns (system, user_msg)."""

def build_prompt_recognition(image_b64, selected_bboxes_with_categories, scene_annotation, image_meta):
    """Task B: bbox -> text. Returns (system, user_msg)."""

def build_prompt_counting_direct(image_b64, category, bboxes, scene_annotation, image_meta):
    """Task C1: direct count. Returns (system, user_msg)."""

def build_prompt_counting_boxed(image_b64, category, bboxes, scene_annotation, image_meta):
    """Task C2: count + list bboxes. Returns (system, user_msg)."""

def build_prompt_counting_verification(image_b64, category, bboxes, scene_annotation, image_meta):
    """Task C3: yes/no threshold question. Returns (system, user_msg)."""

def build_prompt_relational(image_b64, cat_a, bbox_a, cat_b, bbox_b, relation, image_meta):
    """Task D: spatial relation between two objects. Returns (system, user_msg)."""
```

### 5.4 Key module: `task_assigner.py`

```python
def assign_task(sample_id: str, scene_annotation: list) -> dict:
    """Returns task assignment with all needed metadata.

    - Draw from {A: 0.30, B: 0.25, C: 0.25, D: 0.20}
    - For A/C: select one category from scene
    - For B: select 1-3 bboxes (possibly across categories)
    - For C: select one category; sub-draw C1/C2/C3
      - If count < 2: 80% fallback to A, 20% keep as C
    - For D: check eligibility (IoU < 0.1, area < 70%, not hierarchical)
      - If ineligible: fallback to A or B (50/50)
    - Deterministic seed from sample_id
    """
```

### 5.5 Key module: `consolidator.py`

```python
def consolidate_sample(raw_sample: dict) -> dict:
    """Merge multi-turn sample into a scene annotation.

    Returns:
      {
        "image_b64": "...",
        "image_meta": {"width": 1024, "height": 768},
        "sample_id": "5747db47-...",
        "scene": [
          {"category": "House", "bboxes": [[61,55,606,521], ...]},
          {"category": "Tree", "bboxes": [[291,25,382,97], ...]},
          ...
        ]
      }
    """
```

### 5.6 Key module: `validator.py`

Per-task validation rules:

| Check | A | B | C1 | C2 | C3 | D |
|-------|---|---|----|----|-----|---|
| `<START_THINKING>...<END_THINKING>` present | Yes | Yes | Yes | Yes | Yes | Yes |
| No legacy `<ref>` / `<box>` tags | Yes | Yes | Yes | Yes | Yes | Yes |
| All GT bboxes in answer (exact match) | Yes | -- | -- | Yes | -- | Yes |
| Bboxes from question echoed in answer | -- | Yes | -- | -- | -- | -- |
| Count in answer matches GT | -- | -- | Yes | Yes | Yes | -- |
| No bboxes in answer | -- | -- | Yes | -- | -- | -- |
| Yes/no consistent with count vs threshold | -- | -- | -- | -- | Yes | -- |
| Spatial relation consistent with coordinates | -- | -- | -- | -- | -- | Yes |

Failed validation: retry once with the same prompt, then discard.

### 5.7 Pipeline execution

- Processes one shard at a time (or a range via CLI).
- Uses `asyncio` + `httpx` for concurrent API calls (configurable, e.g. 64 in-flight).
- Writes one output JSONL per input shard.
- Logs progress: shard number, samples processed, pass/fail/retry counts per task type.
- CLI: `python pipeline.py --shards 0-100 --api-url http://... --concurrency 64 --output-dir /path/`

### 5.8 Output format

Each output line follows the **exact same schema** as the input JSONL. The image is kept as a `data:image/jpeg;base64,...` URL in the User turn's contents. Scene annotation and task type are stored in the top-level `metadata` for debugging/filtering.

```json
{
  "metadata": {
    "mm_id": "336b7aab7c15ebd845019fa297bc99dac0f72d8d9680280be09c98ef30ba96ca",
    "task_type": "D",
    "scene_annotation": [
      {"category": "Tree", "bboxes": [[3, 27, 436, 478]]},
      {"category": "Land vehicle", "bboxes": [[11, 385, 999, 744]]}
    ],
    "validation_passed": true
  },
  "id": "new-uuid",
  "source": "nemotron_vlm_v2_oi_bbox_2_v1_recaptioned",
  "split": "train",
  "version": "3.0.3",
  "preamble": {
    "platform_preamble": null,
    "safety_preamble": {"mode": "contextual"},
    "default_preamble": {"mode": "interactive"},
    "developer_preamble": null,
    "react_config": {"tools": [], "reasoning_effort": "OFF", "grounding_style": "OFF"},
    "current_timestamp": null,
    "reasoning_clause_in_default_preamble": false
  },
  "turns": [
    {
      "metadata": {},
      "role": "User",
      "contents": [
        {
          "metadata": {"width": 1024, "height": 768},
          "image_url": "data:image/jpeg;base64,...",
          "mime_type": "image/jpeg",
          "language_id": null,
          "content_type": "image"
        },
        {
          "metadata": null,
          "text": "What is the spatial relationship between the tree and the car?",
          "language_id": null,
          "grounding_spans": null,
          "content_type": "text"
        }
      ]
    },
    {
      "metadata": {},
      "role": "Chatbot",
      "contents": [
        {
          "metadata": null,
          "text": "<START_THINKING>The tree is at [3,27,436,478]...<END_THINKING>\nThe tree <|box_start|>[3, 27, 436, 478]<|box_end|> is above and to the left of the car <|box_start|>[11, 385, 999, 744]<|box_end|>.",
          "language_id": null,
          "grounding_spans": null,
          "content_type": "text"
        }
      ]
    }
  ]
}
```

Key schema decisions:
- **All `qwen3_0.6b_emb` embeddings dropped**: These are text embeddings (no image embeddings exist in this dataset). Since the text changes, they're invalid. Both turns have `metadata: {}`. Text/image embeddings can be recomputed downstream if needed.
- **`mm_id` preserved** from the original sample for traceability.
- **`task_type`, `scene_annotation`, `validation_passed`** live in top-level `metadata`.
- **`source` updated** to `nemotron_vlm_v2_oi_bbox_2_v1_recaptioned` to distinguish from the original.
- **`preamble` copied verbatim** from the original sample.

---

## 6. Qwen Prompts (Full Text)

### 6.1 Task A -- Grounding: System Prompt

```
You are an expert visual grounding assistant. You will be shown an image along with a
scene annotation listing all detected object categories and their bounding boxes. You
will be asked to generate a Q&A turn about ONE specific category.

Your job is to produce a high-quality, natural-sounding grounding conversation turn
consisting of:
1. A USER question asking where the object(s) of the specified category are in the image.
2. A CHATBOT answer that first contains a reasoning block enclosed in
   <START_THINKING>...<END_THINKING>, and then lists the bounding boxes.

Rules:
- The bounding boxes are given to you as ground truth in normalized [0, 1000] format.
  You MUST use them exactly as provided -- do not change, reorder, or invent any
  coordinates.
- Wrap every bounding box with the special tokens:
  <|box_start|>[x1, y1, x2, y2]<|box_end|>
- Do NOT use <ref></ref> or <box></box> tags.
- The reasoning block should describe what you observe in the image that leads you to the
  answer. Be specific about visual appearance, position, and context. Keep it concise
  (2-4 sentences).
- Vary the question phrasing naturally. Do not always use the same template.
- Use correct English grammar. Pay attention to singular vs. plural agreement.
- When the category name is a compound noun (e.g., "Human face", "Land vehicle"), use
  natural phrasing in the question.
```

### 6.2 Task B -- Recognition: System Prompt

```
You are an expert visual recognition assistant. You will be shown an image along with a
scene annotation and a selection of bounding box regions to describe.

Your job is to produce a high-quality, natural-sounding recognition conversation turn
consisting of:
1. A USER question that provides the bounding box coordinates and asks what objects are
   at those locations.
2. A CHATBOT answer that first contains a reasoning block enclosed in
   <START_THINKING>...<END_THINKING>, and then describes what is found at each location.

Rules:
- The bounding boxes are given in normalized [0, 1000] format. Wrap every bounding box
  with: <|box_start|>[x1, y1, x2, y2]<|box_end|>
- Do NOT use <ref></ref> or <box></box> tags.
- The user question should embed the bounding boxes and ask about their contents.
- In the reasoning block, describe what you actually see in each region of the image --
  appearances, colors, context, spatial relationships. Be specific and visually grounded.
  Keep it concise (2-4 sentences).
- The final answer should identify each object with a brief natural-language description
  that goes beyond just repeating the category name. Mention distinctive visual attributes
  (color, size, pose, material, etc.) where possible.
- Use correct English grammar throughout.
- The descriptions must be consistent with the ground-truth category labels provided.
  Do not contradict them.
```

### 6.3 Task C1 -- Direct Counting: System Prompt

```
You are an expert visual counting assistant. You will be shown an image along with a
scene annotation. You will be asked to count instances of ONE specific category.

Your job is to produce a high-quality, natural-sounding counting conversation turn
consisting of:
1. A USER question asking how many instances of the category are in the image.
2. A CHATBOT answer that first contains a reasoning block enclosed in
   <START_THINKING>...<END_THINKING>, and then states the count.

Rules:
- The count is given to you as ground truth. You MUST state the exact count provided --
  do not change it.
- Do NOT include any bounding boxes in the answer. Only state the count.
- Do NOT use <ref></ref> or <box></box> tags.
- The reasoning should walk through the counting process -- describe scanning the image
  and identifying each instance. Keep it concise (2-4 sentences).
- Vary the question phrasing naturally.
- Use correct English grammar with proper singular/plural agreement.
```

### 6.4 Task C2 -- Box-based Counting: System Prompt

```
You are an expert visual counting assistant. You will be shown an image along with a
scene annotation. You will be asked to count instances of ONE specific category and
provide their locations.

Your job is to produce a high-quality, natural-sounding counting conversation turn
consisting of:
1. A USER question asking how many instances of the category are in the image and where
   they are located.
2. A CHATBOT answer that first contains a reasoning block enclosed in
   <START_THINKING>...<END_THINKING>, and then states the count followed by a numbered
   list of all bounding boxes.

Rules:
- The count and bounding boxes are given to you as ground truth. You MUST use them
  exactly -- do not change the count or invent/omit any bounding boxes.
- Wrap every bounding box with: <|box_start|>[x1, y1, x2, y2]<|box_end|>
- Do NOT use <ref></ref> or <box></box> tags.
- The reasoning should walk through the counting process -- describe scanning the image
  and identifying each instance. Keep it concise (2-4 sentences).
- Vary the question phrasing naturally.
- Use correct English grammar with proper singular/plural agreement.
```

### 6.5 Task C3 -- Verification Counting: System Prompt

```
You are an expert visual counting assistant. You will be shown an image along with a
scene annotation. You will be asked a yes/no question about whether there are
more/fewer/exactly N instances of a specific category.

Your job is to produce a high-quality, natural-sounding verification conversation turn
consisting of:
1. A USER question asking whether there are more than, fewer than, or exactly N instances
   of the category.
2. A CHATBOT answer that first contains a reasoning block enclosed in
   <START_THINKING>...<END_THINKING>, and then confirms or denies the claim, stating the
   actual count.

Rules:
- The count is given to you as ground truth. You MUST use it exactly.
- A threshold N is also provided. Vary the question format:
  - "More/fewer than N": compare the actual count against N.
  - "Exactly N": check whether the actual count equals N.
- Do NOT include bounding boxes in the answer.
- Do NOT use <ref></ref> or <box></box> tags.
- The reasoning should walk through the counting process -- describe scanning the image
  and counting each instance. Keep it concise (2-4 sentences).
- Vary the question phrasing naturally.
- Use correct English grammar with proper singular/plural agreement.
```

### 6.6 Task D -- Relational: System Prompt

```
You are an expert visual spatial reasoning assistant. You will be shown an image along
with a scene annotation and two specific objects (each with their category and bounding
box). The bounding boxes are spatially separated.

Your job is to produce a high-quality, natural-sounding spatial reasoning conversation
turn consisting of:
1. A USER question asking about the spatial relationship between the two objects.
2. A CHATBOT answer that first contains a reasoning block enclosed in
   <START_THINKING>...<END_THINKING>, and then states the spatial relationship.

Rules:
- The bounding boxes are given in normalized [0, 1000] format. Wrap every bounding box
  with: <|box_start|>[x1, y1, x2, y2]<|box_end|>
- Do NOT use <ref></ref> or <box></box> tags.
- In the reasoning block, describe the positions of both objects and explain how you
  determined their spatial relationship. Keep it concise (2-4 sentences).
- The answer must include both bounding boxes with special tokens.
- The spatial relationship must be consistent with the actual coordinates provided.
- Vary the question format:
  - Yes/no: "Is the X to the right of the Y?"
  - Open-ended: "What is the spatial relationship between the X and the Y?"
  - Comparison: "Which is larger, the X or the Y?"
  - Relative position: "Describe where the X is relative to the Y."
- Use correct English grammar. Refer to the objects with natural language (e.g., "the
  tree" not "the Tree").
```

### 6.7 User Message Template

The user message provides the full scene annotation as context, plus the specific task metadata:

```
Image: {image}

Full scene annotation for this image:
{scene_annotation_formatted}

--- Task ---
Task type: {A|B|C1|C2|C3|D}
{task_specific_fields}

Generate a natural Q&A turn. Output strictly in this JSON format:
{
  "question": "...",
  "answer": "<START_THINKING>...<END_THINKING>\n..."
}
```

Where `{task_specific_fields}` varies by task:
- **A**: `Target category: {category}\nTarget bboxes: {bbox_list}`
- **B**: `Bboxes to describe: {bbox_list_with_categories}`
- **C1/C2/C3**: `Target category: {category}\nGT count: {count}\nGT bboxes: {bbox_list}\nSub-type: {C1|C2|C3}`
- **D**: `Object 1: {cat_a} at {bbox_a}\nObject 2: {cat_b} at {bbox_b}\nGT spatial relation: {relation}`

### 6.8 Constraining the output

- Tasks A and C: bboxes are hard-constrained. The prompt provides them as ground truth and instructs the model to copy them verbatim. Post-processing verifies with regex matching.
- Task B: bboxes appear in the user question (not generated by the model), so they are inherently constrained. The model's category description is soft-validated against the ground-truth label.
- Task D: bboxes and the spatial relation are derived deterministically from coordinates. Post-processing verifies the relation stated in the answer is consistent.

---

## 7. Tracking

| Step | Status |
|------|--------|
| Inspect data & assess bbox quality | DONE |
| Analyze multi-turn structure | DONE |
| Write recaptioning plan | DONE |
| Write Qwen recaptioning prompts | DONE |
| Implement pipeline code | TODO |
| Test end-to-end on sample shard | TODO |
| Run full recaptioning (1024 shards) | TODO |
| Quality filtering pass | TODO |
| Final assembly into multimodal_ift | TODO |

---

## 8. Open Questions for Discussion

1. **Reasoning trace depth**: Should reasoning be 2-4 sentences (current plan) or shorter/longer? Longer reasoning costs more tokens but gives richer training signal.
2. **Task weight tuning**: The 30/25/25/20 split is a starting point. Should we adjust after inspecting a pilot batch?
