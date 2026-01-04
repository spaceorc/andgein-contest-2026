import glob
import os
import shutil
import subprocess
import sys
from io import BytesIO

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import uniform_filter1d
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, CLIPProcessor, CLIPModel

load_dotenv()

URL = "https://2026.andgein.ru/api/tasks/lego"
HEADERS = {"Key": os.environ["ANDGEIN_API_KEY"]}

# Model IDs
DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Reference embeddings for classification (computed from known examples)
lego_centroid = None
not_lego_centroid = None

# Load models once at startup
print("Loading Grounding DINO model...")
processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID)
print("DINO loaded.")

print("Loading CLIP model...")
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, use_fast=True)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
print("CLIP loaded.")


def _get_image_embedding(img: Image.Image) -> np.ndarray:
    """Get normalized CLIP image embedding."""
    inputs = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].numpy()


def _compute_centroids():
    """Compute centroids from reference images."""
    global lego_centroid, not_lego_centroid

    not_lego_files = glob.glob("refs/not_lego/*.jpg")
    lego_files = glob.glob("refs/lego/*.jpg")

    if len(not_lego_files) < 3 or len(lego_files) < 3:
        print(f"Not enough reference images ({len(lego_files)} LEGO, {len(not_lego_files)} not-LEGO).")
        return

    print(f"Computing centroids from {len(lego_files)} LEGO + {len(not_lego_files)} not-LEGO references...")

    not_lego_embs = [_get_image_embedding(Image.open(f)) for f in not_lego_files]
    lego_embs = [_get_image_embedding(Image.open(f)) for f in lego_files]

    not_lego_centroid = np.mean(not_lego_embs, axis=0)
    not_lego_centroid = not_lego_centroid / np.linalg.norm(not_lego_centroid)

    lego_centroid = np.mean(lego_embs, axis=0)
    lego_centroid = lego_centroid / np.linalg.norm(lego_centroid)

    print("Centroids ready.")


_compute_centroids()


def download_image(image_url: str) -> Image.Image:
    """Download image and return PIL Image."""
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))


def split_grid(img: Image.Image, rows: int, cols: int) -> list[list[Image.Image]]:
    """Split image into grid of cells."""
    w, h = img.size
    cw, ch = w // cols, h // rows

    grid = []
    for row in range(rows):
        row_cells = []
        for col in range(cols):
            cell = img.crop((col * cw, row * ch, (col + 1) * cw, (row + 1) * ch))
            row_cells.append(cell)
        grid.append(row_cells)
    return grid


def box_area(box):
    """Calculate area of a bounding box [x1, y1, x2, y2]."""
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_area(box1, box2):
    """Calculate intersection area of two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def filter_overlapping_boxes(boxes_scores, threshold=0.8):
    """
    Remove boxes that overlap >=threshold with a larger box.
    But if a large box would "swallow" 2+ boxes that are closer to median size,
    remove the large box instead.
    """
    if not boxes_scores:
        return []

    # First pass: do normal merge to get median
    sorted_boxes = sorted(boxes_scores, key=lambda x: -box_area(x[0]))
    first_pass = []
    for box, score in sorted_boxes:
        dominated = False
        for kept_box, _ in first_pass:
            inter = intersection_area(box, kept_box)
            box_a = box_area(box)
            if box_a > 0 and inter / box_a >= threshold:
                dominated = True
                break
        if not dominated:
            first_pass.append((box, score))

    if not first_pass:
        return []

    # Calculate median area from first pass result
    areas = [box_area(box) for box, _ in first_pass]
    median_area = sorted(areas)[len(areas) // 2]

    def distance_to_median(box):
        return abs(box_area(box) - median_area)

    # For each box, find which smaller boxes it would dominate
    dominates = {i: [] for i in range(len(boxes_scores))}

    for i, (box1, _) in enumerate(boxes_scores):
        for j, (box2, _) in enumerate(boxes_scores):
            if i == j:
                continue
            area1 = box_area(box1)
            area2 = box_area(box2)
            if area1 <= area2:
                continue
            inter = intersection_area(box1, box2)
            if area2 > 0 and inter / area2 >= threshold:
                dominates[i].append(j)

    # Find boxes to remove: large boxes that dominate 2+ boxes closer to median
    to_remove = set()
    for i, dominated_indices in dominates.items():
        if len(dominated_indices) >= 2:
            box_i = boxes_scores[i][0]
            dist_i = distance_to_median(box_i)
            closer_count = sum(
                1 for j in dominated_indices
                if distance_to_median(boxes_scores[j][0]) < dist_i
            )
            if closer_count >= 2:
                to_remove.add(i)

    # Now do normal filtering, but skip boxes marked for removal
    remaining = [(i, box, score) for i, (box, score) in enumerate(boxes_scores) if i not in to_remove]
    remaining = sorted(remaining, key=lambda x: -box_area(x[1]))

    result = []
    for _, box, score in remaining:
        dominated = False
        for kept_box, _ in result:
            inter = intersection_area(box, kept_box)
            box_a = box_area(box)
            if box_a > 0 and inter / box_a >= threshold:
                dominated = True
                break
        if not dominated:
            result.append((box, score))
    return result


def is_lego_minifigure(crop: Image.Image) -> tuple[bool, dict[str, float]]:
    """Check if cropped image is a LEGO minifigure using CLIP embeddings."""
    emb = _get_image_embedding(crop)
    sim_lego = float(np.dot(emb, lego_centroid))
    sim_not = float(np.dot(emb, not_lego_centroid))
    return sim_lego > sim_not, {"sim_lego": sim_lego, "sim_not_lego": sim_not}


def filter_non_lego(img: Image.Image, boxes_scores: list) -> tuple[list, list]:
    """Filter out non-LEGO figures using CLIP classification."""
    kept = []
    rejected = []
    for box, score in boxes_scores:
        x1, y1, x2, y2 = [int(c) for c in box]
        crop = img.crop((x1, y1, x2, y2))
        is_lego, clip_scores = is_lego_minifigure(crop)
        if is_lego:
            kept.append((box, score))
        else:
            rejected.append((box, score, clip_scores))
    return kept, rejected


def find_figure_boxes(img: Image.Image) -> list[tuple[list, float]]:
    """Find LEGO minifigure boxes in image using Grounding DINO."""
    inputs = processor(images=img, text="LEGO minifigure.", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"], threshold=0.15, target_sizes=[img.size[::-1]]
    )[0]

    # Filter out boxes that are too large (> 15% of image area)
    img_area = img.size[0] * img.size[1]
    boxes_scores = []
    for box, score in zip(results["boxes"], results["scores"]):
        b = box.tolist()
        if box_area(b) / img_area <= 0.15:
            boxes_scores.append((b, score.item()))

    return filter_overlapping_boxes(boxes_scores, threshold=0.8)


def detect_minifigures(cell: Image.Image) -> tuple[int, list, list]:
    """Detect minifigures in a cell using Grounding DINO + CLIP filtering."""
    boxes_scores = find_figure_boxes(cell)

    if boxes_scores:
        kept, rejected = filter_non_lego(cell, boxes_scores)
        boxes = [box for box, _ in kept]
        rejected_boxes = [(box, clip_scores) for box, _, clip_scores in rejected]
    else:
        boxes = []
        rejected_boxes = []

    return len(boxes), boxes, rejected_boxes


def draw_detections(cell: Image.Image, boxes: list) -> Image.Image:
    """Draw detection boxes on image."""
    img_copy = cell.copy()
    draw = ImageDraw.Draw(img_copy)
    for i, box in enumerate(boxes):
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1] - 15), str(i + 1), fill="red")
    return img_copy


def get_grid_size(img: Image.Image) -> tuple[int, int]:
    """Return (rows, cols) grid size by analyzing image gradients."""
    w, h = img.size

    # Compute gradient magnitude (detail map)
    gray = np.array(img.convert('L'), dtype=float)
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    gradient = np.sqrt(gx**2 + gy**2)

    # 1D profiles
    profile_x = gradient.sum(axis=0)
    profile_y = gradient.sum(axis=1)

    def find_best_cells(profile, size, min_cells=3, max_cells=7):
        """Find best number of cells by testing regularity of splits."""
        smoothed = uniform_filter1d(profile, size=50)
        best_score = float('inf')
        best_n = None

        for n_cells in range(min_cells, max_cells + 1):
            cell_size = size / n_cells
            if cell_size < 500 or cell_size > 1200:
                continue

            expected = [int(cell_size * i) for i in range(1, n_cells)]
            window = int(cell_size * 0.2)

            actual_splits = []
            for exp in expected:
                lo = max(0, exp - window)
                hi = min(len(smoothed), exp + window)
                local_min = lo + np.argmin(smoothed[lo:hi])
                actual_splits.append(local_min)

            gaps = np.diff([0] + actual_splits + [size])
            regularity = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 1
            valley_score = sum(smoothed[min(p, len(smoothed)-1)] for p in actual_splits)
            valley_score /= len(actual_splits) if actual_splits else 1
            combined_score = valley_score + regularity * 50000

            if combined_score < best_score:
                best_score = combined_score
                best_n = n_cells

        return best_n or 4

    return find_best_cells(profile_y, h), find_best_cells(profile_x, w)


def solve_with_image(img: Image.Image, grid_override: tuple[int, int] | None = None) -> int:
    """Split image into grid, analyze each cell, return total count."""
    n_rows, n_cols = grid_override if grid_override else get_grid_size(img)

    img.save("lego_current.jpg")
    print(f"Splitting into {n_rows}x{n_cols} grid...")
    grid = split_grid(img, n_rows, n_cols)

    for row in range(n_rows):
        for col in range(n_cols):
            grid[row][col].save(f"lego_cell_{row}_{col}.jpg")

    print("Detecting minifigures with DINO + CLIP...")

    if os.path.exists("rejected"):
        shutil.rmtree("rejected")
    os.makedirs("rejected", exist_ok=True)

    total_cells = n_rows * n_cols
    counts = [[0] * n_cols for _ in range(n_rows)]
    total_rejected = 0
    rej_idx = 0

    for row in range(n_rows):
        for col in range(n_cols):
            cell_num = row * n_cols + col + 1
            print(f"[{cell_num}/{total_cells}] Cell [{row},{col}]... ", end="", flush=True)

            count, boxes, rejected = detect_minifigures(grid[row][col])
            counts[row][col] = count
            total_rejected += len(rejected)

            detected_img = draw_detections(grid[row][col], boxes)
            detected_img.save(f"lego_detected_{row}_{col}.jpg")

            for box, clip_scores in rejected:
                x1, y1, x2, y2 = [int(c) for c in box]
                crop = grid[row][col].crop((x1, y1, x2, y2))
                diff = clip_scores["sim_lego"] - clip_scores["sim_not_lego"]
                crop.save(f"rejected/rej_{rej_idx}_{row}_{col}_diff{diff:+.3f}.jpg")
                rej_idx += 1

            if rejected:
                print(f"{count} (rejected {len(rejected)} non-LEGO)")
            else:
                print(f"{count}")

    # Print results table
    print("\nResults:")
    print("| |" + "".join(f" Col {i} |" for i in range(n_cols)))
    print("|---" + "|---" * n_cols + "|")
    for i, row in enumerate(counts):
        print(f"| Row {i} |" + "".join(f" {c} |" for c in row))

    total = sum(sum(row) for row in counts)
    print(f"\nTOTAL: {total}")
    if total_rejected:
        print(f"(Rejected {total_rejected} non-LEGO detections)")

    return total


def open_detection_images(n_rows: int, n_cols: int):
    """Open all detection images in Preview for manual verification."""
    files = [f"lego_detected_{row}_{col}.jpg" for row in range(n_rows) for col in range(n_cols)]
    files.append("lego_current.jpg")
    subprocess.run(["open", "-a", "Preview"] + files)


def main():
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv

    # Parse --grid NxM argument
    grid_override = None
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--grid="):
            parts = arg.split("=")[1].split("x")
            grid_override = (int(parts[0]), int(parts[1]))
        elif arg == "--grid" and i + 1 < len(sys.argv):
            parts = sys.argv[i + 1].split("x")
            grid_override = (int(parts[0]), int(parts[1]))

    response = requests.get(URL, headers=HEADERS)
    if response.status_code != 200:
        print("Task completed or error:", response.text)
        return

    task = response.json()
    level = task["current_level"]
    image_url = task["parameters"]["image_url"]

    print(f"\n{'='*50}")
    print(f"Level {level}/{task['levels_count']}")
    print(f"Max attempts: {task['max_attempts_count']}")
    if dry_run:
        print("DRY RUN MODE - will not submit")
    print(f"{'='*50}")

    print("Downloading image...")
    img = download_image(image_url)
    print(f"Image size: {img.size[0]}x{img.size[1]}")

    if grid_override:
        n_rows, n_cols = grid_override
        print(f"Grid: {n_rows} rows x {n_cols} cols (OVERRIDE)")
    else:
        n_rows, n_cols = get_grid_size(img)
        print(f"Grid: {n_rows} rows x {n_cols} cols")

    answer = solve_with_image(img, grid_override)

    if dry_run:
        print(f"\nDRY RUN: Would submit answer: {answer}")
        return

    print(f"\nSubmitting answer: {answer}")
    r = requests.post(URL, headers=HEADERS, json={"level": level, "answer": str(answer)}).json()

    if not r["is_correct"]:
        print(f"WRONG! {r.get('checker_output', r)}")
        print("\nOpening detection images for manual verification...")
        open_detection_images(n_rows, n_cols)
        return

    print(f"Level {level} CORRECT! Score: {r.get('new_score', '?')}")


if __name__ == "__main__":
    main()
