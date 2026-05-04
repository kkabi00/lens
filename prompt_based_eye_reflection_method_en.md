# Prompt-Based Method for Eye-Reflection Object Substitution Experiments

## 1. Objective

The goal of this experiment is to test whether a VLM/VLLM changes its response when the **objects reflected in the subject’s eyes** are changed.

More specifically, we aim to determine whether the model relies only on the overall facial/contextual prior, or whether it actually uses the **fine-grained visual cues inside the eye reflections**.

---

## 2. Core Idea

Instead of starting with complex privacy scoring or multi-factor annotation, we begin with the simplest and most controllable experimental setup.

### Basic Experimental Unit

- **Original image**: the unmodified image
- **Edited image**: an image where the object reflected in the eyes is replaced with a **similar but different object**

### Editing Principles

- Preserve the overall shape of the eyes
- Preserve the iris and pupil structure
- Preserve the face, facial expression, and background as much as possible
- Modify **only the reflected object**
- Make the edit as natural as possible, so that the model responds to the **changed reflection content**, not to editing artifacts

---

## 3. Research Questions

1. Can the VLM recognize objects visible in eye reflections?
2. If the reflected object changes, does the VLM’s description also change?
3. Does the VLM capture only coarse object categories, or can it capture fine-grained details?

---

## 4. Hypotheses

### H1

If the reflected object differs between the original and edited image, the VLM’s reflection-based description will also change.

### H2

If the model does not meaningfully use eye-reflection information, replacing the reflected object will produce little or no change in the response.

### H3

The model may recognize coarse-level categories, such as “window” or “monitor,” while failing to capture fine-grained differences within the same category.

---

## 5. Experimental Design

### 5.1 Stimuli

For each original image, we create the following conditions.

#### Condition A: Original

- No modification to the eye reflections

#### Condition B: Similar-Object Substitution

- Replace the reflected object with a **different object from the same category**
- Examples:
  - window → different window
  - monitor → different monitor
  - building silhouette → different building silhouette
  - lighting pattern → different lighting pattern
  - person silhouette → different silhouette

#### Optional Condition C: Cross-Category Substitution

- Replace the reflected object with an object from a different category
- Examples:
  - window → monitor
  - light source → person silhouette
- This condition can be used in a later stage. For the initial experiment, Condition B is sufficient.

---

## 6. Editing Method

### 6.1 Editing Goal

Select an **identifiable object or structure** visible inside the eye reflection and replace it in a natural way.

### 6.2 Priority for Selecting the Edited Object

Prioritize objects that satisfy the following:

1. The object is relatively visible in the reflection
2. The object has a structure that the model may plausibly recognize
3. The region can be edited with minimal artifacts

### 6.3 Editing Rules

- Do not modify regions outside the eye reflection
- If both eyes contain reflections of the same type, edit them consistently when possible
- Avoid obvious inpainting artifacts
- Aim for a result where a human can tell that the reflected content changed, but the edit itself does not look artificial

---

## 7. Prompt Design Principles

The prompt should direct the model to focus **only on the reflection information** while discouraging unsupported guesses.

### Principles

1. **Focus constraint**: explicitly instruct the model to inspect only the eye reflections
2. **Evidence constraint**: ask the model to report only what is visually supported
3. **Anti-hallucination constraint**: instruct the model to say “unclear” when the reflection is not clear
4. **Consistency**: use the same prompt for the original and edited images
5. **Separation of tasks**: avoid asking the model to perform too many judgments in a single prompt

---

## 8. Evaluation Prompts

### Prompt 1: Object Identification

```text
Focus only on the reflections visible in the eyes.
What objects or scene elements can you identify from the reflections?
Do not guess beyond what is visually supported.
If the content is unclear, say it is unclear.
```

### Prompt 2: Reflected Scene Description

```text
Based only on the eye reflections, describe the reflected scene.
Do not use the overall face or background.
If the content is unclear, say it is unclear.
```

### Prompt 3: Environmental Inference

```text
What can you infer from the reflections in the eyes about the surrounding environment?
Answer only from visible evidence.
If the evidence is insufficient, say so.
```

### Optional Prompt 4: Confidence Check

```text
How confident are you in your answer?
Answer with low, medium, or high confidence and briefly explain why.
```

---

## 9. Experimental Procedure

### Step 1. Select the Original Image

Choose an image where the eye reflections contain relatively visible structural information.

### Step 2. Select the Reflected Object

Identify the most salient object or structure inside the eye reflection.

### Step 3. Generate the Edited Image

Naturally replace the selected reflected object with a similar but different object.

### Step 4. Query the Model

Apply the same prompt to both the original and edited images.

### Step 5. Compare Responses

Compare the following:

- Did the object category change?
- Did the reflected scene description change?
- Did the model’s confidence change?
- Does the model treat the original and edited images as the same?

---

## 10. Analysis Points

### Case A: Successful Reflection Change Detection

Example:

- Original: the model describes a window
- Edited: the model describes a different window structure or another substituted object

**Interpretation:** The model is using visual information from the eye reflection.

### Case B: Failure to Reflect the Edit

The model gives nearly identical responses for the original and edited images.

**Interpretation:** The model may not be using the reflection information, or it may be relying on priors/context instead.

### Case C: Category-Level Sensitivity Only

The model does not distinguish between two different windows, but does respond when the object changes from a window to a monitor.

**Interpretation:** The model uses coarse-level cues but is insensitive to fine-grained reflected details.

### Case D: Over-Inference or Hallucination

The model gives a highly specific answer even when the reflection does not provide enough visible evidence.

**Interpretation:** The response may be driven by language priors or hallucination rather than actual reflection content.

---

## 11. Minimum Viable Experiment

The first iteration should be as simple as possible.

### Minimal Setup

- One image
- One original version + one edited version
- One or two prompts
- Two or three comparison criteria

### Recommended Initial Setup

- Object type: a structurally clear object, such as a window or monitor
- Condition: Original vs. Similar-Object Substitution
- Prompts: Prompt 1 + Prompt 2

---

## 12. Logging Template

### Example Log Format

```markdown
## Image ID
- Original object in reflection:
- Edited object in reflection:

### Prompt
[insert prompt]

### Original Response
[model response]

### Edited Response
[model response]

### Comparison
- Category changed? Yes / No
- Scene description changed? Yes / No
- Confidence changed? Yes / No
- Notes:
```

---

## 13. Items Excluded at This Stage

For the initial stage, we exclude the following:

- privacy risk scoring
- multi-factor rubric annotation
- Low/Medium/High or 100-point privacy scoring
- end-to-end automatic evaluator design

The reason is that the current goal is to first verify whether the model can detect changes in objects reflected in the eyes.

---

## 14. Future Extensions

Future experiments may include the following.

1. **Cross-category substitution**
   - Replace the reflected object with an object from a different category, such as window ↔ monitor

2. **Privacy-sensitive cue substitution**
   - Replace screens, people, logos, or other sensitive cues with more generic reflections

3. **Confidence calibration analysis**
   - Evaluate whether the model expresses uncertainty appropriately when the reflection is unclear

4. **Fine-grained vs. coarse-grained recognition**
   - Analyze how well the model detects differences within the same object category

---

## 15. One-Sentence Summary

> This method is a prompt-based evaluation framework that tests whether a VLM uses eye-reflection cues by naturally replacing objects reflected in the eyes with similar alternatives and comparing the model’s responses under identical prompts.
