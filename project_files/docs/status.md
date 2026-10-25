# Status

Last updated: 2026-10-25

## Now
Phase 0 (footage and ground truth) is finishing. Phase 1 has not started.

## Done
- Planning and tool research (see plan.md)
- Test photos taken from floor 20, 1x and 2x. 2x chosen because the junction fills the frame
- Capture settings decided (see plan.md, Phase 0)

## Next task
Set up the repo and environment, then build Phase 1 on a short clip:
1. Create .venv and pyproject.toml, install ultralytics, supervision, opencv-python, pandas, pyarrow, pyyaml, pytest
2. Read the clip, run the detector, run ByteTrack, write the trajectory parquet
3. Render an annotated debug video
4. Check ID stability by eye before moving on

## Open questions (blockers marked)
- BLOCKER: does the owner have a GPU, or Azure / Colab access? Decides feasible resolution and speed
- iPhone model (decides whether the 2x lens is a real sensor crop)
- Is the signal fixed-time or actuated? Answer comes from the timing log
- Are night or low-light recordings in scope? Assumed no for now
- Real-time live feed or offline only? Assumed offline only
- Licence: ultralytics is AGPL-3.0, raise with the supervisor before any productisation

## Measured facts (fill in, do not guess)
- Clip file name:
- Resolution / fps (from ffprobe):
- Duration:
- Signal cycle length(s):
- Fixed-time or actuated:
