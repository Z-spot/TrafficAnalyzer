# Traffic Flow Analysis

Offline computer vision pipeline that detects and tracks vehicles at one signalised 4-way intersection from a fixed, elevated camera (about floor 20), then turns the tracks into turning movement counts and traffic KPIs.

## Current scope (read this first)

We are building Phases 1 to 4 only: detect, track, calibrate to ground coordinates, count movements, compute KPIs.
SUMO simulation and reinforcement learning are Phases 5 and 6. They are out of scope until the owner says otherwise. Do not build ahead, and do not add SUMO or RL dependencies.

Current status and next task: @docs/status.md

Full roadmap, concepts and rationale live in docs/plan.md. Read it on demand when a task needs it, it is not imported here on purpose.

## Who you are working with

The owner (Zane) is a cybersecurity graduate, comfortable with Python and ML, new to computer vision video pipelines and traffic engineering. Explain new CV or traffic concepts briefly when you introduce them. Be direct: if an approach has a flaw, say so before building it.

## Stack

- Python, managed with uv or pip in a local .venv (see docs/decisions.md for the final choice)
- ultralytics for detection (YOLO). Model name is set in config/pipeline.yaml, never hardcoded
- supervision (Roboflow) for Detections objects, ByteTrack, zones, annotators, video IO
- opencv-python for homography (cv2.getPerspectiveTransform / findHomography)
- pandas + pyarrow for trajectory tables, matplotlib for charts, PyYAML for config
- Exact versions are pinned in pyproject.toml. Do not guess APIs from memory: these libraries change fast, check the installed version and its docs before using a function you are not sure about.
- Licence note: ultralytics is AGPL-3.0. Do not add other dependencies without asking.

## Layout

- src/trafficflow/  pipeline code (detect, track, calibrate, zones, movements, kpis, cli)
- config/  every scene-specific number lives here (camera.yaml, zones.yaml, pipeline.yaml)
- scripts/  small interactive helpers (point picker, zone picker)
- data/raw/ video (gitignored), data/interim/ trajectory parquet, data/processed/ counts and KPIs
- data/interim/clips/  dev video cuts (gitignored)
- data/interim/detections/  cached detector output (gitignored), not the trajectory schema
- docs/  plan.md, status.md, decisions.md, glossary.md
- notebooks/  exploration only, never pipeline logic

## Hard rules

1. No scene-specific numbers in code. Pixel coordinates, homography points, zone polygons, fps, class IDs, thresholds all come from config/.
2. The trajectory table schema below is a contract. Do not change it without updating docs/decisions.md and this file in the same change.
3. Anchor point for any ground-plane maths is the bottom centre of the bounding box, never the box centre.
4. Never edit anything in data/raw/. Video files are originals.
5. Pipeline logic goes in src/, as functions that can be called from tests. Notebooks only call them.
6. Do not add a dependency without asking.
7. Never read frame rate from assumptions. Read it from the video file and store it with the output.
8. Finish and commit one phase before starting the next. Ask before starting a new phase.
9. timestamp_s comes from the container's presentation timestamps, never from frame_idx divided by an assumed fps.
10. Never estimate speed or delay from a single frame interval. Frame spacing varies by about 5 percent, so smooth over several frames.

## Trajectory schema (Parquet, one row per detection per frame)

frame_idx (int), timestamp_s (float), track_id (int), class_id (int), class_name (str), conf (float),
x1, y1, x2, y2 (float, pixels), anchor_x, anchor_y (float, pixels, bottom centre of box),
world_x, world_y (float, metres, added in Phase 2, null before calibration)

## Commands

Fill these in as the CLI is built. Keep this list accurate.

- Set up: `uv sync` (Python 3.12). The venv lives OUTSIDE the project because ~/Documents is synced by iCloud, which evicts files and hangs Python: `export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/trafficflow` before every uv command, and use `uv run --no-sync`
- Whole chain on a frame range: `python -m trafficflow.cli run --name NAME --start-frame N --end-frame M` (detect, track with road mask, smooth, render raw and smoothed debug videos). Files go to data/interim/
- One stage from cached files: `cli detect` (slow, GPU, caches all detections), then `cli track`, `cli smooth`, `cli render` (seconds each, with `--name NAME`)
- Draw the road mask by clicking (opens a window, run it in your own Terminal): `python scripts/pick_road.py`. Starts empty and replaces the saved mask. `--keep` starts from the saved polygons
- Track stats to compare outputs: `python scripts/track_stats.py a.parquet b.parquet`
- Tests: `pytest -q`

## Working style

- Use plan mode for anything that touches more than one module, and show the plan before editing.
- Start Phase 1 on a short clip (2 to 5 minutes), not the full recording.
- After finishing a task, update docs/status.md (what is done, what is next, open questions) and log any non-obvious choice in docs/decisions.md with the reason.
- Say exactly what is uncertain. Do not invent library functions, model names or numbers.
- Keep explanations short and plain. No filler.
