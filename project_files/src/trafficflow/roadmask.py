"""Road mask: one or more polygons in full-frame pixels. A point is on the road if it is inside any of them."""

import cv2
import numpy as np

from trafficflow.config import load_config


def load_road_polygons() -> list[np.ndarray]:
    """Polygons from config/zones.yaml as a list of (N, 2) float32 arrays. Empty list if none drawn yet.

    Accepts a list of polygons, or a single flat polygon (list of [x, y]).
    """
    raw = load_config("zones.yaml").get("road_mask") or []
    if raw and not isinstance(raw[0][0], (list, tuple)):
        raw = [raw]
    return [np.array(p, dtype=np.float32) for p in raw if len(p) >= 3]


def points_inside(points: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    """Boolean per (x, y) point: inside or on any polygon."""
    out = np.zeros(len(points), dtype=bool)
    for poly in polygons:
        contour = poly.reshape(-1, 1, 2)
        for i, (x, y) in enumerate(points):
            if not out[i]:
                out[i] = cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0
    return out


def inside_mask(xyxy: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    """Boolean per box: is the bottom-centre (rule 3) inside the road."""
    anchors = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]])
    return points_inside(anchors, polygons)


def write_road_mask(zones_path, polygons: list[list[tuple[float, float]]]) -> None:
    """Rewrite the road_mask key of zones.yaml as a list of polygons, keeping everything above it."""
    s = open(zones_path).read()
    marker = "# Road mask:"
    head = s[: s.index(marker)] if marker in s else s.rstrip("\n") + "\n\n"
    out = [head + "# Road mask: only vehicles whose bottom-centre is inside these polygons (for at least\n"
           "# tracker.min_inside_fraction of a track's frames) are kept. Full-frame pixels (3840x2160).\n"
           "# Written by scripts/pick_road.py. A list of polygons, each a list of [x, y].\n"]
    out.append("road_mask:\n" if polygons else "road_mask: []\n")
    for i, poly in enumerate(polygons, 1):
        out.append(f"  - # polygon {i}\n")
        out.extend(f"    - [{round(x)}, {round(y)}]\n" for x, y in poly)
    open(zones_path, "w").write("".join(out))
