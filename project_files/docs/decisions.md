# Decisions log

Newest at the bottom. Format: date, decision, reason, what would make us revisit it.

## 2026-10-25 Initial decisions

- Offline processing of recorded video, not live. Reason: far simpler, and nothing in phases 1 to 4 needs real time. Revisit if a live demo is requested.
- Detector: Ultralytics YOLO, pretrained COCO weights first, no training. Model name lives in config/pipeline.yaml. Start with a medium size model, raise imgsz before reaching for tiled inference. Revisit if far-queue cars are missed.
- Tracker: ByteTrack first. Switch to BoT-SORT only if ID switches are measured to be a problem.
- Ground plane: homography from 4 or more surveyed points, applied only to tracked bottom-centre anchor points, not to whole frames.
- Movements: entry and exit zone pairs per track, not line crossings alone.
- Camera: 4K, constant frame rate (25 fps via Blackmagic Camera app), H.264, stabilisation off, locked focus and white balance, 2x lens. Frames for calibration are extracted from the video, not from stills, because stills are 4:3 and video is 16:9. (fps and codec corrected below: actual footage is 30 fps HEVC.)
- Signal timing is inferred from vehicle discharge in the footage (effective green), not read from the lamps. State this in any report.
- Out of scope for now: SUMO calibration (Phase 5) and RL (Phase 6).

## 2026-10-25 Corrections after recording

- Frame rate is 30 fps, not 25. Codec is HEVC (hvc1), not H.264. Reason: measured with ffprobe on data/raw/main_video.MOV (r_frame_rate 30/1, average about 30.0057 fps). config/pipeline.yaml expected_fps updated to 30. Revisit if a later session is recorded with different settings.
- Timestamps come from container presentation timestamps, not frame_idx / fps. Reason: frame gaps are 19 to 21 ticks on a 1/600 s clock (31.7 to 35.0 ms), about 5 percent variation, so assuming a fixed fps drifts and single-frame speeds are noisy.

## 2026-10-25 Environment (Phase 1, Task 1)

- Package manager: uv (0.13.5, Homebrew). Reason: lockfile with transitive pins, manages its own Python, `exclude-newer` support. Revisit if uv causes friction.
- Python 3.12 (uv-managed, 3.12.15), not the system 3.14. Reason: longest-tested version for the CV stack, and numpy 2.5 needs 3.12 or newer. Revisit when a dependency drops 3.12.
- Version policy: only releases at least two weeks old. `[tool.uv] exclude-newer = 2026-10-11`, exact `==` pins for direct dependencies, uv.lock committed.
- Pins: ultralytics 8.4.175, supervision 0.30.9, opencv-python 4.14.0.94, pandas 3.0.6, pyarrow 26.0.0, pyyaml 6.0.3, pytest 9.1.1 (dev group). Transitive: torch 2.14.1, torchvision 0.29.1, numpy 2.5.4.
- OpenCV held at 4.x. Reason: an unpinned resolve picks OpenCV 5.0, a new major version. 4.14 is the maintained 4.x line. Revisit once ultralytics and supervision state support for 5.
- `YOLO_AUTOINSTALL=False` is set in `trafficflow/__init__.py` before ultralytics is imported. Reason: ultralytics otherwise pip-installs missing packages at run time, which breaks the rule against unasked dependencies.
- Tracker implementation: `sv.ByteTrack` from supervision 0.30.9. It is deprecated (removed in supervision 0.31) and its replacement is the separate `trackers` package, a new dependency. Using the deprecated class for now to avoid adding one unasked. Revisit before upgrading supervision past 0.30.x.
- Timestamps: `build_pts_index` reads packet PTS with ffprobe, sorts them (B-frames), and gives timestamp_s per frame_idx. Tracker frame_rate is the container's avg_frame_rate. No sidecar file for now, it takes seconds to compute.
- Reference PNG names (frame_000200 etc.) are sample numbers, not video frame indices. They match video frames 3601, 9002 and 21605.

## 2026-10-26 Flicker, parked cars, environment location

- Measured on frames 3600-3750 before changing anything: box position wobble was small (0.15-0.3 px on stationary vehicles). The visible flicker came from boxes dropping out (11 tracks with holes, 214 missing rows), class labels flipping (8 of 74 tracks) and short ghost tracks.
- detector.conf 0.25 -> 0.10. supervision's ByteTrack matches low-score detections (0.1 up to track_activation_threshold) in a second stage, and conf 0.25 starved it. New tracks still need score >= 0.25 plus 0.1.
- Pipeline split into stages: detect (slow, cached to data/interim/detections/), track (road mask + ByteTrack), smooth, render. Reason: tuning mask and tracker no longer needs a 2 minute detector run. The detection cache is not the trajectory schema.
- Road mask: `road_mask` in config/zones.yaml, a LIST of polygons in full-frame pixels (a point is on the road if inside any). Superseded by the 2026-10-27 entry below on how it is applied. A polygon on the image does this job without homography. Homography stays in Phase 2 (needs measured distances). Revisit if cars parked on the road itself matter: add a per-track "never moved" flag in Phase 3/4.
- Smoothing is offline (uses past and future frames): one class per track by confidence-weighted vote, gaps up to 10 frames interpolated (conf is NaN on those rows), tracks under 15 frames dropped, centred rolling mean on box centre (7) and size (15). Output keeps the trajectory schema, written to *.smooth.parquet. Raw trajectories are kept.
- The venv lives at ~/.venvs/trafficflow (UV_PROJECT_ENVIRONMENT), not in the project. Reason: ~/Documents is synced by iCloud Drive, which evicted 7,012 of 26,594 venv files and hung imports. The 8.8 GB video is in the same synced folder and could be evicted too. Revisit by moving the project out of ~/Documents or turning off Desktop and Documents sync.

## 2026-10-27 Road mask applied per track, drawn with a picker

- The mask is now a track-level rule, not a per-detection filter. Everything is tracked, then a track is kept if its bottom-centre is on the road for at least `tracker.min_inside_fraction` (0.5) of its frames. Reason: dropping detections outside a coarse polygon cut tracks at curved exits and kerbs, so cars popped in and out. Cost: parked cars still use tracker IDs internally before being dropped.
- The polygons are drawn by clicking with `scripts/pick_road.py` (several polygons allowed), not by hand-written coordinates. Reason: the first draft (21 points) missed curved exits and included parked cars at the top.

## 2026-10-27 Single zone, track stitching

- zones.yaml now holds only the owner's clicked polygon. The old 22-point draft was removed after measuring: without it 0 of 115 tracks were dropped and 0.4% of rows lay outside. `pick_road.py` now starts empty and replaces the saved mask; `--keep` loads the existing polygons as a starting point.
- Vanishing boxes are not caused by the polygon edge: 99.1% of confident detections inside the polygon already reached the output. The cause is the detector missing a vehicle (hidden, small), ByteTrack giving up after 30 frames, and a new ID starting. Re-clicking for kerb margin does not fix this.
- Added offline track stitching to the smooth stage. A track end is linked to a later track start if same class, restart within 60 frames, near the position predicted by velocity from both sides (tolerance 60 px + 3 px per frame), mutual best match, and no rival present at the same time within max(1.5x error, 25 px). Holes inside tracks are then filled up to 60 frames. On the 30 s run: 21 links, ID-switch candidates 5 -> 1, tracks with holes 6 -> 0, tracks under 1 s 10 -> 3. Revisit if links are found to merge different vehicles (every link is printed by `cli smooth`).
- `tracker.lost_track_buffer` stays 30: 45 and 60 gave essentially the same tracks on the 30 s run.
- Homography does not bring back an occluded vehicle. It can later improve motion prediction for stitching (metres instead of pixels). Phase 2.

## 2026-10-27 v3 run: larger zone

- A wider 32-point zone (up to the top edge of the frame) did not make the late-starting top-of-frame cars start earlier (checked track 94: still starts at frame 3996). Those cars are seen by the detector 18-119 frames earlier at confidence 0.10-0.25, but ByteTrack needs >= 0.35 to start a track (`track_activation_threshold` 0.25 + 0.1). Next experiment: lower the start threshold, measured per vehicle with strict matching. A quick test suggested 14-18 frames gained for some cars but raised stitched links from 21 to 36-45, so the wrong-merge risk must be measured too.
- The wider zone also pulled in 7 parked cars (the bottom row of the car park, x 444-876, y 216-259 in 1080p, never moving). The upper-left edge needs to stay below that row. Not a code change: re-click that edge.

## To decide
- matplotlib: direct pin, or rely on it arriving through ultralytics
- Git root: keep at the parent folder, or move it into project_files/
- Model size (m, l or x): after the Task 2 benchmark
