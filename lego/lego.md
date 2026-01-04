# LEGO Minifigure Counting Task - Report

## Task Description

**Competition:** andgein.ru 2026
**Task:** Count LEGO minifigures in images
**Levels:** 45 levels with increasing complexity
**Final Score:** 3528

The task required downloading images from an API, splitting them into grid cells, detecting LEGO minifigures in each cell, and submitting the total count. The challenge was distinguishing actual LEGO minifigures from other objects (toys, figurines, decorations) that might appear similar.

## Solution Pipeline

```
Image → Grid Split → Object Detection (DINO) → Box Filtering → LEGO Classification (CLIP) → Count
```

### 1. Grid Auto-Detection

Images were divided into grids of varying sizes (3x3 to 8x8). We implemented automatic grid detection using gradient analysis:

- Compute [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator) gradient magnitude (detail map)
- Sum gradients along X and Y axes to get 1D profiles
- Find valleys at expected grid line positions
- Score candidates by regularity and valley depth
- Select grid with most square-like cells

Manual override available via `--grid NxM` parameter when auto-detection fails.

### 2. Object Detection with Grounding DINO

**Library:** [`transformers`](https://pypi.org/project/transformers/) (Hugging Face)
**Model:** [`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny)

Grounding DINO is a [zero-shot](https://en.wikipedia.org/wiki/Zero-shot_learning) object detection model that finds objects based on text prompts. We used:
- Prompt: `"LEGO minifigure."`
- Detection threshold: 0.15

**Key features:**
- No training required - works out of the box
- Text-guided detection allows flexible object specification
- Returns [bounding boxes](https://en.wikipedia.org/wiki/Minimum_bounding_box) with confidence scores

### 3. Box Filtering

Raw detections required filtering:

**Size filtering:**
- Max box ratio: 15% of cell area (removes oversized detections)

**Overlap merging (Smart algorithm):**
1. First pass: merge overlapping boxes (threshold 80%)
2. Compute median area from first pass results
3. Second pass: if a large box "swallows" 2+ boxes closer to median size, remove the large box instead

This prevents large false detections from eliminating multiple valid smaller figures.

### 4. LEGO Classification with CLIP

**Library:** [`transformers`](https://pypi.org/project/transformers/) (Hugging Face)
**Model:** [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)

CLIP (Contrastive Language-Image Pre-training) creates [embeddings](https://en.wikipedia.org/wiki/Word_embedding) for images and text in a shared vector space. We used it to distinguish LEGO from non-LEGO figures.

**Approach: Reference [Centroid](https://en.wikipedia.org/wiki/Centroid) Classification**

Instead of text prompts (which were unstable), we used reference image embeddings:

1. Collect reference images:
   - `refs/lego/` - 18 confirmed LEGO minifigure crops
   - `refs/not_lego/` - 11 confirmed non-LEGO objects

2. Compute centroids:
   ```python
   lego_centroid = normalize(mean([embed(img) for img in lego_refs]))
   not_lego_centroid = normalize(mean([embed(img) for img in not_lego_refs]))
   ```

3. Classify by [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity):
   ```python
   sim_lego = dot(embedding, lego_centroid)
   sim_not_lego = dot(embedding, not_lego_centroid)
   is_lego = sim_lego > sim_not_lego
   ```

**Why this works better than text prompts:**
- More stable classification boundary
- Easy to improve by adding misclassified examples to references
- Handles edge cases (Jar Jar Binks, unusual LEGO figures)

## Libraries Used

### Grounding DINO
- **Purpose:** Zero-shot object detection
- **Paper:** "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"
- **Key idea:** Combines DINO (self-supervised [vision transformer](https://en.wikipedia.org/wiki/Vision_transformer)) with grounded pre-training to enable text-guided detection
- **Advantage:** No fine-tuning needed, works with any text description

### CLIP (OpenAI)
- **Purpose:** Image-text similarity / image classification
- **Paper:** "Learning Transferable Visual Models From Natural Language Supervision"
- **Key idea:** Train image and text encoders jointly using [contrastive learning](https://en.wikipedia.org/wiki/Contrastive_learning) on 400M image-text pairs so they share embedding space
- **Advantage:** Zero-shot classification via similarity to reference embeddings

### Other Dependencies
- [`Pillow`](https://pypi.org/project/pillow/) - Image manipulation
- [`torch`](https://pypi.org/project/torch/) - PyTorch backend for models
- [`numpy`](https://pypi.org/project/numpy/) - Numerical operations
- [`scipy`](https://pypi.org/project/scipy/) - Gradient computation for grid detection
- [`requests`](https://pypi.org/project/requests/) - API communication

## Key Challenges Solved

| Problem | Solution |
|---------|----------|
| Grid size varies per level | Auto-detection via gradient analysis + manual override |
| Large boxes swallow small valid figures | Smart merge: preserve boxes closer to median size |
| CLIP text prompts unstable | Reference centroid approach with example images |
| Borderline LEGO figures rejected | Add to `refs/lego/` to shift classification boundary |
| Non-LEGO objects accepted | Add to `refs/not_lego/` to improve discrimination |

## Results Summary

All 45 levels completed successfully with iterative refinement of reference images and box filtering algorithm.
