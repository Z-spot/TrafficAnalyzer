# Project plan: Traffic Flow Analysis Tool

Owner: Zane. Written 2026-10-25 from the planning conversation. Update this file when the plan changes and log why in decisions.md.

## 1. Goal and scope

Build an offline tool that turns fixed-camera video of one signalised intersection into trajectories, turning movement counts and traffic KPIs.

Long-term vision from the supervisor (keep in mind, do not build yet):
1. Collect data to build an accurate simulation
2. Turn that into a model and add reinforcement learning to the signal control
3. Seek improvements in traffic KPIs

This is a multi-month research arc. Phases 1 to 4 are the deliverable now. Phases 5 and 6 are the long arc.

## 2. The scene (from the owner's photos, floor 20)

- Four approaches, each split into 3 lane groups
- Not a textbook 4-way: slip lanes and a curving ramp bottom right. The real movement set must be mapped by hand before counting logic is written
- At 2x zoom the junction fills the frame. North and south queues run off the top and bottom, west is partly hidden by the owner's own building on the left. Queue length will be partial on every approach
- Signal heads at the near (south) stop line should face the camera but are tiny. Check in a test clip whether lamp colour is readable
- A tower crane crosses the right edge of the east approach and moves
- Two small figures stood in the middle of the junction in one photo: pedestrians or traffic police. If police direct traffic, signal timing from those periods is not usable

## 3. Concepts (short)

- Detection finds boxes per frame. Tracking links them into IDs over time. Track IDs are what make movement counting possible.
- Homography maps image points on the road plane to metres. Apply it only to the bottom centre of each tracked box. Warping whole frames to a top-down video is optional and only for demos.
- Zones: a movement is an (entry zone, exit zone) pair per track. supervision gives PolygonZone and LineZone. LineZone needs tracker IDs.
- A signal cycle is one pass through all phases. Fixed-time signals repeat exactly. Actuated signals vary with demand.

## 4. Stack

| Layer | Tool |
|---|---|
| Detection | ultralytics YOLO (latest family at time of build, model name in config) |
| Tracking | ByteTrack (via supervision or ultralytics), BoT-SORT only if needed |
| Zones, annotation, video IO | supervision |
| Geometry | opencv-python |
| Data | pandas, pyarrow |
| Plots | matplotlib |
| Config | YAML |
| Phase 5 | SUMO, TraCI |
| Phase 6 | sumo-rl, stable-baselines3 |

Known facts to respect: ultralytics is AGPL-3.0 (commercial licence sold separately). Raise with the supervisor before any productisation.

Reference material found during planning:
- Roboflow tutorial on vehicle speed estimation with YOLO, ByteTrack and perspective transform (good template for Phases 1 and 2)
- supervision "Count in Zone" guide (PolygonZone, LineZone)
- Published work on ground-plane projection for turning movement counts at intersections (back-projection improved trajectory classification over image-plane analysis)
- SUMO-RL, built on SUMO, with RESCO benchmark networks (Phase 6 starting point)

## 5. Phases

### Phase 0: Footage and ground truth (in progress)

Capture settings: 4K, 25 fps constant, H.264, Rec. 709, stabilisation off, focus and white balance locked, 2x lens, shutter around 1/250 (watch white car roofs for clipping), lens correction automatic, time-of-day timecode on. Tripod on taped marks, same framing every session. Lights off behind the glass, lens close to the window, check for ghost reflections.

Deliverables:
- Main clip 15 to 20 minutes, constant frame rate confirmed with ffprobe (compare r_frame_rate and avg_frame_rate)
- Detector sanity test on a frame extracted from the video: are far-end cars detected?
- 3 to 5 reference frames extracted from the video for calibration and zones
- Signal timing table from the footage, at least 6 consecutive cycles
- Clip log: date, time, weather, anything unusual

Signal timing method: scrub the footage frame by frame. Per approach and cycle log discharge start (first queued vehicle moves), discharge end (last vehicle crosses the stop line), and next conflicting discharge start. Convert frames to seconds with the real fps. Constant cycle length within about 2 s suggests fixed-time, otherwise actuated. Look for saw-cut loop outlines near the stop lines. This measures effective green, not the lamp.

### Phase 1: Detect and track, export trajectories (3 to 5 days)

Deliverable is the trajectory Parquet file, the annotated video is a debugging aid.
1. Read video, detect per frame, keep vehicle classes (car, bus, truck, motorcycle)
2. ByteTrack for track IDs
3. Write one row per detection per frame using the schema in CLAUDE.md
4. Render an annotated debug video

Known risk: small distant vehicles. Try in order: larger imgsz (for example 1280), crop to an ROI around the junction, then tiled inference (SAHI style) only if needed.
Exit: a car crossing the junction keeps one ID most of the time. Fix ID flicker before moving on.

### Phase 2: Calibration and world coordinates (2 to 4 days)
1. Pick 4 or more widely spread, fixed, identifiable ground points on an extracted video frame (scripts/pick_points.py)
2. Get real distances from satellite imagery measurements, standard road markings, kerb or pole spacing measured safely
3. Compute the homography, store points and matrix in config/camera.yaml
4. Add world_x, world_y to the trajectories
5. Validate on a held-out known point and report error in metres. Warp one frame to check lane markings come out parallel
6. Produce a top-down plot of all tracks

### Phase 3: Movement classification and counts (3 to 5 days)
1. Map the real movement set by hand, slip lanes included
2. Draw entry and exit polygons (scripts/pick_zones.py), save to config/zones.yaml
3. Assign each complete track a movement
4. Handle and log bad tracks: enter without exit, appear mid-junction, never leave. Report the rejection rate
5. Aggregate counts per movement per 15 minute bin
6. Validate against a manual count of one 15 minute period

### Phase 4: Traffic KPIs (3 to 5 days)
Throughput per movement and bin, queue length per approach (note off-frame truncation), stopped delay, travel time through the junction, saturation flow, lane utilisation, signal phase and cycle. Deliver as a report with charts.

### Phase 5: SUMO calibration (out of scope for now)
Build the network in netedit or import from OpenStreetMap. Phase 3 counts become flow rates, observed signal timing becomes the signal program. Validate simulated queues and delay against Phase 4 measurements and tune driver behaviour parameters.

### Phase 6: Reinforcement learning (out of scope for now)
sumo-rl Gymnasium environment on the calibrated network, DQN or PPO agent, reward usually negative waiting time or queue, compare against the real fixed-time baseline. Better in simulation is not the same claim as better on the street.

## 6. How much footage

- Development: 2 to 5 minutes, replayed hundreds of times
- Validation: 15 to 30 minutes with manual ground truth counts
- Simulation calibration: several 1 to 2 hour recordings across morning peak, midday and evening peak, ideally repeated on several weekdays
Do not record hours before Phase 1 works. The framing will probably change.

## 7. Risks

- Occlusion: buses hide cars, palms and poles cut tracks, expect ID switches
- Camera drift: any movement invalidates the homography. Lock the mount, reverify per session
- Lighting: hard shadow lines, glare, and night is a separate problem
- Stationary vehicles in queues are a classic tracker failure. Check this case specifically, queue length depends on it
- Pretrained COCO model is generic. If accuracy is poor, fine-tune on a few hundred annotated frames from this scene
- Compute: a GPU is strongly preferred for hours of 4K footage. Unknown at time of writing, tracked in status.md
- Scope creep: hold the line on Phases 1 to 4 and show working output early
