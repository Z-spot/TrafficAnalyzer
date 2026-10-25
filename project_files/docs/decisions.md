# Decisions log

Newest at the bottom. Format: date, decision, reason, what would make us revisit it.

## 2026-10-25 Initial decisions

- Offline processing of recorded video, not live. Reason: far simpler, and nothing in phases 1 to 4 needs real time. Revisit if a live demo is requested.
- Detector: Ultralytics YOLO, pretrained COCO weights first, no training. Model name lives in config/pipeline.yaml. Start with a medium size model, raise imgsz before reaching for tiled inference. Revisit if far-queue cars are missed.
- Tracker: ByteTrack first. Switch to BoT-SORT only if ID switches are measured to be a problem.
- Ground plane: homography from 4 or more surveyed points, applied only to tracked bottom-centre anchor points, not to whole frames.
- Movements: entry and exit zone pairs per track, not line crossings alone.
- Camera: 4K, constant frame rate (25 fps via Blackmagic Camera app), H.264, stabilisation off, locked focus and white balance, 2x lens. Frames for calibration are extracted from the video, not from stills, because stills are 4:3 and video is 16:9.
- Signal timing is inferred from vehicle discharge in the footage (effective green), not read from the lamps. State this in any report.
- Out of scope for now: SUMO calibration (Phase 5) and RL (Phase 6).

## To decide
- Package manager: uv or pip + venv
- Python version
- Pinned versions of ultralytics, supervision, opencv-python
