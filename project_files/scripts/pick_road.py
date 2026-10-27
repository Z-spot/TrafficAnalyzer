"""Click the road outline on a frame. Saves polygons to config/zones.yaml (road_mask) and a preview image.

Run in your own Terminal (it opens a window):
  export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/trafficflow
  uv run --no-sync python scripts/pick_road.py [--image PATH] [--scale 0.4] [--keep]

By default you start from an EMPTY canvas and saving REPLACES whatever road_mask was in config/zones.yaml.
Use --keep to load the existing polygons as a starting point instead.

Mouse: left click adds a point to the current polygon.
Keys:  Enter = close this polygon and start another   Backspace = undo last point (or last polygon)
       s = save and quit   c = clear everything   q = quit without saving
Trace each arm and each curved exit as its own polygon if that is easier. Follow the kerb closely,
and keep out of the car park and other parked areas. Polygons may overlap.
"""

import argparse

import cv2
import numpy as np

from trafficflow.config import CONFIG_DIR, PROJECT_ROOT
from trafficflow.roadmask import load_road_polygons, write_road_mask

ap = argparse.ArgumentParser()
ap.add_argument("--image", default=str(PROJECT_ROOT / "data/raw/reference_frames/frame_000200.png"))
ap.add_argument("--scale", type=float, default=0.4, help="display scale, 0.4 fits a laptop screen")
ap.add_argument("--keep", action="store_true", help="start from the polygons already in config/zones.yaml")
args = ap.parse_args()

img = cv2.imread(args.image)
if img is None:
    raise SystemExit(f"cannot read {args.image}")
scale = args.scale
base = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
existing = load_road_polygons()
done: list[list[tuple[float, float]]] = [[tuple(pt) for pt in p.tolist()] for p in existing] if args.keep else []
current: list[tuple[float, float]] = []
WIN = "road outline  (Enter=next polygon, Backspace=undo, s=save, q=quit)"


def to_disp(poly):
    return (np.array(poly, dtype=np.float32) * scale).astype(np.int32).reshape(-1, 1, 2)


def draw(show_current=True) -> np.ndarray:
    canvas = base.copy()
    if done:
        union = np.zeros(canvas.shape[:2], np.uint8)
        for p in done:  # one call per polygon: a single call would use even-odd fill and punch holes where they overlap
            cv2.fillPoly(union, [to_disp(p)], 255)
        overlay = canvas.copy()
        overlay[union > 0] = (0, 255, 255)
        canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
        cv2.polylines(canvas, [to_disp(p) for p in done], True, (0, 255, 255), 2)
    if show_current and current:
        pts = to_disp(current)
        cv2.polylines(canvas, [pts], False, (0, 0, 255), 2)
        for p in pts.reshape(-1, 2):
            cv2.circle(canvas, tuple(int(v) for v in p), 3, (0, 0, 255), -1)
    return canvas


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current.append((x / scale, y / scale))  # stored in full-frame pixels


cv2.namedWindow(WIN)
cv2.setMouseCallback(WIN, on_mouse)
print(__doc__)
while True:
    cv2.imshow(WIN, draw())
    k = cv2.waitKey(30) & 0xFF
    if k == 13 and len(current) >= 3:
        done.append(current.copy())
        current.clear()
    elif k in (8, 127):
        if current:
            current.pop()
        elif done:
            done.pop()
    elif k == ord("c"):
        done.clear()
        current.clear()
    elif k == ord("q"):
        print("quit without saving")
        break
    elif k == ord("s"):
        if len(current) >= 3:
            done.append(current.copy())
        write_road_mask(CONFIG_DIR / "zones.yaml", done)
        preview = PROJECT_ROOT / "data/interim/debug/road_mask_preview.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview), draw(show_current=False))
        print(f"replaced {len(existing)} polygon(s) with {len(done)} in config/zones.yaml; preview: {preview}"
              if not args.keep else f"saved {len(done)} polygon(s) (started from {len(existing)}) to config/zones.yaml; preview: {preview}")
        break
cv2.destroyAllWindows()
