# Status

Last updated: 2026-10-27

## Now
Phase 1 in progress. The detect, track, smooth and render chain works on a 150 frame range (3600-3750). The road mask is applied per track (kept if on the road at least half its frames). Road outline clicked with scripts/pick_road.py (2 polygons). A 30 second run (frames 3600-4500) has been reviewed. Not committed yet.

## Done
- Planning and tool research (see plan.md)
- Test photos taken from floor 20, 1x and 2x. 2x chosen because the junction fills the frame
- Capture settings decided (see plan.md, Phase 0)
- Environment: uv, Python 3.12, pinned versions, venv at ~/.venvs/trafficflow (outside iCloud-synced Documents). MPS GPU verified. Frame count matches ffprobe (38,881)
- Pipeline stages: detect (cached) -> track (road mask, ByteTrack) -> smooth -> render. 8 unit tests pass
- Flicker fix on frames 3600-3750: tracks with gaps 11 -> 0, class-flip tracks 8 -> 0, frame-to-frame jitter about 7x lower. Parked car park cars no longer tracked
- 30 s run (frames 3600-4500, 900 frames): detect + decode 0.129 s/frame on the M4 GPU, so about 84 min for the full 38,881 frames with yolo26m at imgsz 1280. 115 smoothed tracks, 21 last the whole 30 s, 5 likely ID switches, 6 tracks with holes over 10 frames, 0 class flips after smoothing

## Known issues / later
- Small residual flicker in some boxes (position and size jitter between frames)
- Motorbikes and riders are mostly missed by the detector
- Cars at the far-left edge of the west arm are partly hidden by the building
- Labels overlap in dense queues
- The parked white pickup on the pavement by the kiosk is still tracked (road polygon includes that corner). Fix by re-clicking with scripts/pick_road.py
- Vehicles hidden by others for long stretches still lose their box. Stitching bridges up to 60 frames. Stronger model (l or x) or homography based prediction (Phase 2) may help
- Skill `codebase-to-course` installed in .claude/skills/. The upstream repo has no LICENSE file. Not run yet. .claude/ is not gitignored

- Active zone restored to the v2 44-point polygon (2026-10-27). The v3 32-point zone is kept in config/zones.v3.yaml. v3 zone pulls in the bottom row of the car park (7 parked cars). Re-click the upper-left edge below that row
- Top-of-frame cars start late because ByteTrack needs score >= 0.35 to start a track (detector sees them earlier at 0.10-0.25). Not fixed by a bigger zone

## Next task
1. Review the stitched links by eye (run30s_v2_smooth.mp4). Re-click the pavement corner if wanted
2. Speed benchmark of yolo26m vs l vs x, and a far-end detection count on one frame
3. Full video run, then T10: close Phase 1 and commit (ask first)

## Open questions (blockers marked)
- (answered) GPU: Apple M4 with MPS, works. About 0.13 s/frame
- (answered) iPhone 15 Pro, 2x lens tag 48mm
- Is the signal fixed-time or actuated? Answer comes from the timing log
- Are night or low-light recordings in scope? Assumed no for now
- Real-time live feed or offline only? Assumed offline only
- Licence: ultralytics is AGPL-3.0, raise with the supervisor before any productisation

## Measured facts (fill in, do not guess)
- Laptop: Mac
- Clip file: data/raw/main_video.MOV (exact case, do not rename)
- Shot 2026-10-25 on an iPhone 15 Pro at 2x (lens tag "iPhone 15 Pro 48mm", f/1.78), Blackmagic Camera app
- Clip start timecode: 10:34:04 local time (Kuwait, UTC+3), Sunday mid-morning
- ffprobe: 3840x2160, HEVC (hvc1), yuv420p, bt709, 8 bit, about 54 Mbps, one video stream
- Frame rate: r_frame_rate 30/1, average about 30.0057 fps
- Frames / duration: 38,881 frames, 1295.785 s
- Frame timing: timestamps on a 1/600 s clock. Gap between consecutive frames is 19, 20 or 21 ticks (31.7 to 35.0 ms). No dropped frames
- Capture settings: shutter 1/250, ISO 100, white balance 4000K, stabilisation off
- Dev clip: data/interim/clips/test_60s.mov, video only, stream-copy cut near 5:00 of the original. 1815 frames over 60.098 s, probably because it starts on an earlier keyframe. Not frame-exact, benchmarking only
- Reference frames: data/raw/reference_frames/
- Model preference: yolo26m.pt, or l or x if speed allows
- Signal cycle length(s):
- Fixed-time or actuated:
