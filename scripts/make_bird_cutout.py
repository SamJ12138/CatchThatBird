"""One-off: cut the bird out of the public-domain source photo and save the
RGBA PNG that make_synth_video.py composites (data/samples/assets/).

    python scripts/make_bird_cutout.py path/to/House_sparrow_(female)_(50108651573).jpg

Source and license: data/samples/assets/CREDITS.md. The committed PNG is the
output of this script; it only needs re-running to change the cutout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "samples" / "assets" / "house_sparrow_female.png"
# In the 3000x2003 original: the bird (with its feet on the railing).
CROP_XYXY = (960, 680, 1960, 1340)
BIRD_IN_CROP_XYWH = (40, 40, 920, 600)  # GrabCut's "probably foreground" box
WIDTH = 320  # output width in pixels


def cutout(photo: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = CROP_XYXY
    crop = photo[y0:y1, x0:x1].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    cv2.grabCut(crop, mask, BIRD_IN_CROP_XYWH, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Keep the largest blob (the bird), fill small holes, soften the edge.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = np.where(labels == largest, 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    alpha = cv2.GaussianBlur(fg, (7, 7), 0)
    x, y, w, h = cv2.boundingRect(fg)
    rgba = np.dstack([crop, alpha])[y:y + h, x:x + w]
    scale = WIDTH / rgba.shape[1]
    return cv2.resize(rgba, (WIDTH, round(rgba.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("photo", type=Path)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    photo = cv2.imread(str(args.photo))
    if photo is None or photo.shape[:2] != (2003, 3000):
        raise SystemExit(f"{args.photo}: expected the 3000x2003 source photo (see CREDITS.md)")
    rgba = cutout(photo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), rgba)
    print(f"Wrote {args.out} ({rgba.shape[1]}x{rgba.shape[0]} RGBA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
