"""Step 1: turn the Roboflow COCO export into masks plus a manifest with source groups.

Why the manifest matters: the 200 images are Roboflow exports where every source frame
appears 3 times (augmentations), and 158 images come from ONE video. Any random split
therefore leaks near-duplicates between train and validation. The `frame` and `video`
columns let every later script split by source instead of by file.

Masks: the raw polygon masks are saved to --raw_out untouched; the cleaned (smoothed) masks go to
--out and are what every later script reads. See clean_mask() for what 'cleaned' means.

Usage:
    python src/prepare_data.py --root data/roboflow --out data/masks --raw_out data/masks_raw
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


def frame_id(fname: str) -> str:
    """'Apache_C62_Bladder_mp4-0011_jpg.rf.<hash>.jpg' -> 'Apache_C62_Bladder_mp4-0011'."""
    stem = fname.split(".rf.")[0]
    return re.sub(r"(_(jpg|png))+$", "", stem)


def video_id(frame: str) -> str:
    """'Apache_C62_Bladder_mp4-0011' -> 'Apache_C62_Bladder_mp4' (drops the frame counter)."""
    return re.sub(r"-\d+$", "", frame)


def polygons_to_mask(segmentation, width, height):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    rings = segmentation if isinstance(segmentation[0], list) else [segmentation]
    for ring in rings:
        draw.polygon(ring, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


def clean_mask(mask, sigma=1.5):
    """Smooth a hand-drawn polygon mask without changing its size or shape noticeably.

    1. fill holes and keep only the largest connected piece (drops stray specks),
    2. blur the 0/1 mask with a Gaussian and threshold at 0.5. This rounds off 1-px staircase steps
       and thin spikes while keeping the area almost unchanged (a straight edge does not move).
    """
    m = ndi.binary_fill_holes(mask.astype(bool))
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    smooth = ndi.gaussian_filter(m.astype(np.float32), sigma)
    return (smooth > 0.5).astype(np.uint8)


def main(root: Path, out: Path, raw_out: Path, sigma: float):
    coco = json.load(open(root / "_annotations.coco.json"))
    images = {i["id"]: i for i in coco["images"]}
    by_image = {}
    for ann in coco["annotations"]:
        if isinstance(ann["segmentation"], dict):
            raise RuntimeError("RLE masks are not handled; export polygons from Roboflow.")
        by_image.setdefault(ann["image_id"], []).append(ann)

    out.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)
    rows = []
    for img_id, info in images.items():
        w, h = info["width"], info["height"]
        raw = np.zeros((h, w), dtype=np.uint8)
        for ann in by_image.get(img_id, []):
            if ann["segmentation"]:
                raw = np.maximum(raw, polygons_to_mask(ann["segmentation"], w, h))
        mask = clean_mask(raw, sigma=sigma)
        fname = info["file_name"]
        stem = Path(fname).stem
        Image.fromarray(raw * 255).save(raw_out / (stem + "_mask.png"))
        Image.fromarray(mask * 255).save(out / (stem + "_mask.png"))
        gray = np.array(Image.open(root / fname).convert("L"))
        fr = frame_id(fname)
        rows.append(
            dict(
                file=fname,
                frame=fr,
                video=video_id(fr),
                width=w,
                height=h,
                lumen_px=int(mask.sum()),
                lumen_frac=round(float(mask.mean()), 4),
                area_change=round(float(mask.sum() / max(1, raw.sum()) - 1.0), 4),
                black_frac=round(float((gray < 5).mean()), 3),
            )
        )

    rows.sort(key=lambda r: r["file"])
    with open(out.parent / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} cleaned masks to {out}, raw masks to {raw_out}, manifest to {out.parent / 'manifest.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/roboflow"))
    ap.add_argument("--out", type=Path, default=Path("data/masks"))
    ap.add_argument("--raw_out", type=Path, default=Path("data/masks_raw"))
    ap.add_argument("--sigma", type=float, default=1.5)
    a = ap.parse_args()
    main(a.root, a.out, a.raw_out, a.sigma)
