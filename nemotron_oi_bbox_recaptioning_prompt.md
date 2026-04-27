# Nemotron OI BBox: Qwen Recaptioning Prompts

Below are the system prompts and user-message templates for each of the three task types. In all cases, the Qwen model receives the image and structured metadata extracted from the original dataset. The bboxes are provided as ground truth — the model must use them exactly and must not hallucinate new ones.

---

## Task A — Grounding (text → bbox)

### System prompt

```
You are an expert visual grounding assistant. You will be shown an image along with an object category and the ground-truth bounding boxes for all instances of that category.

Your job is to produce a high-quality, natural-sounding grounding conversation turn consisting of:
1. A USER question asking where the object(s) are in the image.
2. A CHATBOT answer that first contains a reasoning block enclosed in <START_THINKING>...</END_THINKING>, and then lists the bounding boxes.

Rules:
- The bounding boxes are given to you as ground truth in normalized [0, 1000] format. You MUST use them exactly as provided — do not change, reorder, or invent any coordinates.
- Wrap every bounding box with the special tokens: <|box_start|>[x1, y1, x2, y2]<|box_end|>
- Do NOT use <ref></ref> or <box></box> tags.
- The reasoning block should describe what you observe in the image that leads you to the answer. Be specific about visual appearance, position, and context. Keep it concise (2-4 sentences).
- Vary the question phrasing naturally. Do not always use the same template.
- Use correct English grammar. Pay attention to singular vs. plural agreement (e.g., "There is 1 flower" not "There are 1 flower"; "its position" not "their positions" when singular).
- When the category name is a compound noun (e.g., "Human face", "Land vehicle"), use natural phrasing in the question (e.g., "Where are the human faces?" rather than "Where are all the Human face?").
```

### User message template

```
Image: {image}

Category: {category}
Number of instances: {count}
Ground-truth bounding boxes (normalized [0, 1000]):
{bbox_list}

Generate a natural grounding Q&A turn. Output strictly in this JSON format:
{
  "question": "...",
  "answer": "<START_THINKING>...<END_THINKING>\n..."
}
```

### Example input

```
Category: Flower
Number of instances: 2
Ground-truth bounding boxes (normalized [0, 1000]):
1. [463, 488, 528, 631]
2. [818, 198, 852, 293]
```

### Example output

```json
{
  "question": "Can you point out where the flowers are in this image?",
  "answer": "<START_THINKING>Looking at the image, I can identify two flowers. One is a larger flower positioned in the center-left portion of the image, roughly in the lower half. The second is a smaller flower located toward the upper-right area of the scene.<END_THINKING>\nI found 2 flowers in this image:\n1. <|box_start|>[463, 488, 528, 631]<|box_end|>\n2. <|box_start|>[818, 198, 852, 293]<|box_end|>"
}
```

---

## Task B — Recognition (bbox → text)

### System prompt

```
You are an expert visual recognition assistant. You will be shown an image along with one or more bounding box regions and the ground-truth category label for those regions.

Your job is to produce a high-quality, natural-sounding recognition conversation turn consisting of:
1. A USER question that provides the bounding box coordinates and asks what objects are at those locations.
2. A CHATBOT answer that first contains a reasoning block enclosed in <START_THINKING>...</END_THINKING>, and then describes what is found at each location.

Rules:
- The bounding boxes are given in normalized [0, 1000] format. Wrap every bounding box with: <|box_start|>[x1, y1, x2, y2]<|box_end|>
- Do NOT use <ref></ref> or <box></box> tags.
- The user question should embed the bounding boxes and ask about their contents.
- In the reasoning block, describe what you actually see in each region of the image — appearances, colors, context, spatial relationships. Be specific and visually grounded. Keep it concise (2-4 sentences).
- The final answer should identify each object with a brief natural-language description that goes beyond just repeating the category name. Mention distinctive visual attributes (color, size, pose, material, etc.) where possible.
- Use correct English grammar throughout.
- The descriptions must be consistent with the ground-truth category label provided. Do not contradict it.
```

### User message template

```
Image: {image}

Ground-truth category: {category}
Bounding boxes (normalized [0, 1000]):
{bbox_list}

Generate a natural recognition Q&A turn. Output strictly in this JSON format:
{
  "question": "...",
  "answer": "<START_THINKING>...<END_THINKING>\n..."
}
```

### Example input

```
Ground-truth category: Flower
Bounding boxes (normalized [0, 1000]):
1. [463, 488, 528, 631]
2. [818, 198, 852, 293]
```

### Example output

```json
{
  "question": "What objects are located at <|box_start|>[463, 488, 528, 631]<|box_end|> and <|box_start|>[818, 198, 852, 293]<|box_end|>?",
  "answer": "<START_THINKING>Examining the first region at [463, 488, 528, 631], I can see a pink flower in bloom, positioned among green foliage in the lower-center area. The second region at [818, 198, 852, 293] contains a small white flower bud near the top-right of the image.<END_THINKING>\nThe objects at the specified locations are:\n1. <|box_start|>[463, 488, 528, 631]<|box_end|> — A pink flower in bloom surrounded by green leaves.\n2. <|box_start|>[818, 198, 852, 293]<|box_end|> — A small white flower bud."
}
```

---

## Task C — Counting (multi-bbox → count)

### System prompt

```
You are an expert visual counting assistant. You will be shown an image along with an object category, the ground-truth count, and the ground-truth bounding boxes for all instances.

Your job is to produce a high-quality, natural-sounding counting conversation turn. You will be told which counting sub-type to generate:

- **C1 (Direct counting):** The user asks how many X are in the image. The answer gives the count only (no bounding boxes).
- **C2 (Box-based counting):** The user asks how many X are in the image and requests their locations. The answer gives the count and lists all bounding boxes.
- **C3 (Verification counting):** The user asks a yes/no question about whether there are more/fewer than N instances. The answer confirms or denies and gives the actual count.

Rules:
- The count and bounding boxes are ground truth. You MUST use them exactly — do not change the count or invent/omit any bounding boxes.
- Wrap every bounding box with: <|box_start|>[x1, y1, x2, y2]<|box_end|>
- Do NOT use <ref></ref> or <box></box> tags.
- Every answer must start with a reasoning block enclosed in <START_THINKING>...</END_THINKING>. The reasoning should walk through the counting process — describe scanning the image and identifying each instance. Keep it concise (2-4 sentences).
- For C3, choose a threshold N that makes the question non-trivial (not always "yes" or always "no"). A good heuristic: pick N = count ± random(1,3).
- Vary question phrasing naturally.
- Use correct English grammar with proper singular/plural agreement.
```

### User message template

```
Image: {image}

Category: {category}
Ground-truth count: {count}
Ground-truth bounding boxes (normalized [0, 1000]):
{bbox_list}
Counting sub-type: {C1|C2|C3}

Generate a natural counting Q&A turn. Output strictly in this JSON format:
{
  "question": "...",
  "answer": "<START_THINKING>...<END_THINKING>\n..."
}
```

### Example input (C2)

```
Category: Car
Ground-truth count: 3
Ground-truth bounding boxes (normalized [0, 1000]):
1. [50, 400, 600, 900]
2. [650, 300, 750, 450]
3. [800, 320, 880, 430]
Counting sub-type: C2
```

### Example output (C2)

```json
{
  "question": "How many cars can you see in this image? Show me where each one is located.",
  "answer": "<START_THINKING>Scanning the image for cars, I can see a large car dominating the lower-center area, plus two smaller cars parked side by side in the middle-right background. That gives me a total of 3.<END_THINKING>\nThere are 3 cars in this image:\n1. <|box_start|>[50, 400, 600, 900]<|box_end|>\n2. <|box_start|>[650, 300, 750, 450]<|box_end|>\n3. <|box_start|>[800, 320, 880, 430]<|box_end|>"
}
```

### Example input (C3)

```
Category: Person
Ground-truth count: 7
Counting sub-type: C3
```

### Example output (C3)

```json
{
  "question": "Are there fewer than 5 people in this photo?",
  "answer": "<START_THINKING>Let me count the people visible in this photo. I can see several individuals — counting carefully, I find 7 people total spread across the scene. That is more than 5, not fewer.<END_THINKING>\nNo, there are actually 7 people in this photo, which is more than 5."
}
```

---

## Implementation notes

### Batching strategy

Each turn is randomly assigned **exactly one** task type (A: 40%, B: 30%, C: 30%) to keep the dataset size ~1:1 with the source. For multi-turn samples with K turns, this yields K recaptioning requests total (not 3K). Task C falls back to A/B when `count < 2` with probability 80%/20%.

### Constraining the output

The bboxes in Tasks A and C are hard-constrained: the prompt provides them as ground truth and instructs the model to copy them verbatim. Post-processing should verify this with regex matching.

For Task B, the bboxes appear in the user question (not generated by the model), so they are inherently constrained. The model's category description is soft-validated against the ground-truth label.

### Post-processing checklist

- [ ] Parse JSON output; discard malformed responses.
- [ ] Verify `<|box_start|>` / `<|box_end|>` token presence and well-formedness.
- [ ] Verify `<START_THINKING>` / `<END_THINKING>` presence.
- [ ] For Tasks A/C: verify all ground-truth bboxes appear in the answer (exact match).
- [ ] For Task B: run a lightweight semantic similarity check between model description and category label.
- [ ] For Task C: verify the count in the answer matches ground truth.
- [ ] Grammar spot-check on a random sample.
