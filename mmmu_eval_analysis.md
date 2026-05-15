# MMMU Evaluation Analysis: VLMEvalKit vs Bee, Online vs Offline

## Executive Summary

We investigated accuracy discrepancies across MMMU evaluation runs and found two independent issues:

1. **VLMEvalKit vs Bee internal harness (~8pp gap):** VLMEvalKit uses a minimal prompt (`"Please select the correct answer"`) without chain-of-thought instructions, producing verbose unstructured outputs that its own extraction pipeline fails to score. 90% of the gap is a scoring artifact, not a model capability difference. The fix is to enable `custom_prompt = "cohere_cot"` in `twenty_average.toml`.

2. **Online (blobheart) vs offline (vLLM) serving (~4-5pp gap):** The Jinja chat template used by the offline/vLLM pipeline (`cohere2_vllm_plugin` v0.2.14) places `<|IMG_PATCH|>` tokens **inside** `<|START_TEXT|>...<|END_TEXT|>` blocks, while the Cohere Chat API (blobheart) places them **outside** — splitting text blocks around images. The online format matches training data organization. This template mismatch causes the model to process images differently offline, producing genuinely worse answers (not just a scoring difference). All 900 prompts differ between the two backends, 49% of questions get different final answers, and the online backend picks the correct answer more often.

---

## 1. VLMEvalKit vs Bee Internal Harness

### 1.1 Prompt Differences

| Aspect | Bee (`MMMU_Gen.MMMU_COT`) | VLMEvalKit (`VLM_MMMU`) |
|--------|---------------------------|-------------------------|
| **Prompt suffix** | Full COT instruction: `"Analyze the image and question carefully, using step-by-step reasoning... Final Answer: <answer>"` | `"Please select the correct answer from the options above."` |
| **Option format** | `(A) option1 (B) option2 ...` | `A. option1\nB. option2\n...` under `Options:` header |
| **Image placement** | Inline at `<image N>` positions within question text | All images prepended before text |
| **Dataset** | Validation only (900 samples) | Dev + Validation (1050 samples) |

**Source files:**
- Bee: `apiary-main/bee/bee/tasks/library/vision/mmmu/mmmu.py` (task), `bee/bee/task_utils/vision/cot.py` (COT prompt)
- VLMEvalKit: `apiary-main/bee/bee/tasks/library/vision/vlmeval_kit/vlmeval_kit_task.py` (wrapper), upstream VLMEvalKit `build_prompt()`
- Config: `apiary-main/bee/bee/tasks/library/vision/vlmeval_kit/twenty_average.toml` (`custom_prompt = "cohere_cot"` is **commented out**)

### 1.2 Answer Extraction & Scoring

| Aspect | Bee (COT mode) | VLMEvalKit |
|--------|----------------|------------|
| **Primary extraction** | Regex on `"Final Answer:"` (99%+ success rate) | `can_infer()` regex for letter patterns |
| **Fallback** | `GeneralJudgeExtractorMetric` (now `command-a-03-2025-eval`) | GPT-4o-mini judge |
| **MCQ metric** | `mmmu_relaxed_accuracy` (numeric tolerance +/-5%, case-insensitive) | `hit`: binary 0/1 exact letter match |

### 1.3 Accuracy Gap Breakdown

On 908 matched validation questions:

| Run | Accuracy |
|-----|----------|
| Bee (online, best) | **58.6%** |
| VLMEvalKit | **50.6%** |

Of 179 cases where Bee is correct but VLMEvalKit is wrong:
- **162 (90.5%)**: VLMEvalKit generation **contains** the correct answer but extraction failed
- 17 (9.5%): genuinely different model answer

The gap is overwhelmingly a **scoring artifact** caused by the missing COT prompt instruction, not a model capability difference.

---

## 2. Online (Blobheart) vs Offline (vLLM) Serving

### 2.1 The Problem

Four runs using the **same** Bee internal harness task (`MMMU_Gen.MMMU_COT`) on the **same** 900 validation questions produced systematically different accuracy depending on the serving backend:

| Run | Backend | Thinking | Accuracy |
|-----|---------|----------|----------|
| f46 | Blobheart (online) | ON | **58.1%** |
| f46c | Blobheart (online) | Force-OFF | **57.8%** |
| f00 | vLLM (offline) | OFF | **54.6%** |
| f59 | vLLM (offline) | ON | **53.4%** |

The ~4-5pp gap is consistent across thinking ON/OFF variants, ruling out thinking mode as the cause.

### 2.2 Root Cause: Chat Template Mismatch

All 900 raw prompts differ between online and offline backends. The difference is in how `<|IMG_PATCH|>` tokens are placed relative to `<|START_TEXT|>` / `<|END_TEXT|>` boundaries:

**Online (blobheart) — images outside text blocks:**
```
<|START_TEXT|>text before image<|END_TEXT|><|IMG_PATCH|><|START_TEXT|>text after image<|END_TEXT|>
```

**Offline (vLLM) — images inside text blocks:**
```
<|START_TEXT|>text before image<|IMG_PATCH|>text after image<|END_TEXT|>
```

For image-first questions, the same pattern holds: online puts `<|IMG_PATCH|>` before `<|START_TEXT|>`, offline puts `<|START_TEXT|>` first then `<|IMG_PATCH|>` inside.

### 2.3 Why This Happens: Code Trace

The two backends follow entirely different rendering paths to convert the same structured `Turn` objects into token sequences:

**Shared starting point** — both get identical `Turn` objects from:
- `apiary-main/bee/bee/tasks/library/vision/mmmu/mmmu.py` → `_build_turns_from_row()` produces interleaved `TextContent` + `ImageContent` parts

**Online path (blobheart):**
1. `hive/hive/blobheart.py` → `content_to_cohere_format()` converts to structured `{"type":"text"}` / `{"type":"image_url"}` parts
2. Sent to **Cohere Chat API** as structured messages
3. The API's **server-side** C4 formatter applies the chat template, splitting text blocks around images
4. `raw_prompt` is returned via the API's debug event (`responses[0].prompt` or streaming `event.prompt`)
5. No template code exists in `apiary-main` for this path — it's entirely server-side

**Offline path (vLLM):**
1. `hive/hive/estimators/offline/vllm_messages.py` → `_content_to_openai_format()` converts to structured parts
2. `_inject_image_placeholders()` (line 114) adds `{"type":"text", "text":"<|IMG_PATCH|>"}` before each `image_url` block
3. The message list is passed to HuggingFace `tokenizer.apply_chat_template()` with a Jinja template from `cohere2_vllm_plugin.chat_templates` (e.g. `C4_THINKING`)
4. `hive/hive/estimators/offline/ray_data_vllm.py` → `_resolve_chat_template()` loads the template, `setup()` wires it into the Ray pipeline

**The bug is in the Jinja template** (`cohere2_vllm_plugin` v0.2.14, `C4_THINKING`):

The `render_content` macro only handles text items and silently skips `image_url` items:
```jinja
{%- macro render_content(content) -%}
  ...
  {%- for item in content -%}
    {%- if item is string -%}
      {{ item }}
    {%- elif item is mapping and (item.type|default("text"))|lower == "text" and item.text is defined -%}
      {{ item.text }}          {# <-- renders injected "<|IMG_PATCH|>" as plain text #}
    {%- endif -%}              {# <-- image_url items silently dropped #}
  {%- endfor -%}
{%- endmacro -%}
```

And the user message wraps everything in a **single** `<|START_TEXT|>...<|END_TEXT|>` block:
```jinja
<|START_OF_TURN_TOKEN|><|USER_TOKEN|><|START_TEXT|>{{ render_content(message.content) }}<|END_TEXT|>
```

So the injected `<|IMG_PATCH|>` text part is rendered inside the text block, producing `<|START_TEXT|>...<|IMG_PATCH|>...<|END_TEXT|>` instead of splitting around the image.

### 2.4 Training Data Format

Training data stores images and text as **separate structured content items**:
```json
{
  "contents": [
    {"content_type": "image", "image_url": "data:image/jpeg;base64,..."},
    {"content_type": "text", "text": "question about the image"}
  ]
}
```

The dominant pattern across ~600 training datasets is `[image, text]` (image first, then text) or `[text, image]` (text first, then image) as separate content items. The training pipeline converts these to token sequences using the same format as the Cohere Chat API — **images outside text blocks**. The offline vLLM template does not faithfully reproduce this format.

### 2.5 Impact: Genuinely Different Model Answers

This is not a scoring artifact — the model produces genuinely worse answers with the offline template:

- **0/900 generations are identical** between online and offline
- **49% of questions get different final answers** (440/900)
- Both backends produce "Final Answer:" ~99% of the time, so it's not an extraction issue
- Among 134 questions where only online is correct: 105 (78%) show the correct answer mentioned in the offline reasoning but the model picks the wrong final answer
- The online backend has a **net +42 more correct answers** (134 online-only-correct vs 92 offline-only-correct)

### 2.6 Key Files

| File | Role |
|------|------|
| `apiary-main/bee/bee/tasks/library/vision/mmmu/mmmu.py` | Builds `Turn` objects (shared, identical for both backends) |
| `apiary-main/hive/hive/blobheart.py` | Online path: sends structured parts to Chat API, receives `raw_prompt` from server |
| `apiary-main/hive/hive/estimators/offline/vllm_messages.py` | Offline path: `_inject_image_placeholders()` inserts `<|IMG_PATCH|>` as text blocks |
| `apiary-main/hive/hive/estimators/offline/ray_data_vllm.py` | Offline path: loads chat template from `cohere2_vllm_plugin`, wires `apply_chat_template` |
| `cohere2_vllm_plugin.chat_templates` (external, v0.2.14) | Jinja template with `render_content` macro that doesn't handle image boundaries |

---

## 3. Recommendations

1. **Fix the offline chat template** in `cohere2_vllm_plugin`: update `render_content` or the user message template line to emit `<|END_TEXT|><|IMG_PATCH|><|START_TEXT|>` when encountering image content, matching the Cohere Chat API's behavior and training data format. This should recover ~4-5pp on MMMU and likely affects all vision benchmarks run offline.

2. **Enable `custom_prompt = "cohere_cot"` in `twenty_average.toml`** for VLMEvalKit's `VLM_MMMU` task (if VLMEvalKit is still used for MMMU). This is the single highest-impact fix for the VLMEvalKit scoring artifact.

3. **Verify other vision benchmarks**: the `<|START_TEXT|>/<|END_TEXT|>` template issue affects all multimodal tasks run via the offline/vLLM pipeline, not just MMMU. Any benchmark with images should show an accuracy gap between online and offline serving.
