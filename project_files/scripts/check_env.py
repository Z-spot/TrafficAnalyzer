"""Phase 1 Task 1: check the environment can run the pipeline on this machine.

Run from the project root: uv run python scripts/check_env.py [--skip-count]
"""

import argparse
import re
import sys

import cv2
from ultralytics import YOLO

from trafficflow.config import PROJECT_ROOT, load_config
from trafficflow.envcheck import (
    check_decoded_frame_count,
    check_model_classes,
    check_mps,
    check_mps_matches_cpu,
    check_opencv_open,
    check_reference_frames,
)
from trafficflow.video import probe_video_stream

VIDEO = PROJECT_ROOT / "data/raw/main_video.MOV"
REF_DIR = PROJECT_ROOT / "data/raw/reference_frames"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-count", action="store_true", help="skip the full-decode frame count (several minutes)")
    args = ap.parse_args()

    det_cfg = load_config()["detector"]
    refs = {int(re.search(r"(\d+)", p.stem).group(1)): p for p in sorted(REF_DIR.glob("frame_*.png"))}

    results = [check_mps()]
    model = YOLO(det_cfg["model"])
    results.append(check_model_classes(model, det_cfg["vehicle_classes"]))
    results.append(check_mps_matches_cpu(model, cv2.imread(str(refs[max(refs)])), det_cfg))
    probe = probe_video_stream(VIDEO)
    print(f"ffprobe: {probe}")
    results.append(check_opencv_open(VIDEO, (probe["width"], probe["height"])))
    results.append(check_reference_frames(VIDEO, refs))
    if not args.skip_count:
        results.append(check_decoded_frame_count(VIDEO, probe["nb_frames"]))

    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
