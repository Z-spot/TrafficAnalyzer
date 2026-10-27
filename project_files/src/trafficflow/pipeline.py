"""Pipeline stages as functions: detect (slow, cached), track, smooth. See cli.py for the commands."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import supervision as sv

from trafficflow.detect import Detector
from trafficflow.roadmask import points_inside
from trafficflow.smooth import smooth_trajectories
from trafficflow.track import make_tracker
from trafficflow.video import average_fps, build_pts_index, iter_frames, probe_video_stream

DETECTION_COLS = ["frame_idx", "timestamp_s", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2"]
TRAJECTORY_COLS = ["frame_idx", "timestamp_s", "track_id", "class_id", "class_name", "conf",
                   "x1", "y1", "x2", "y2", "anchor_x", "anchor_y", "world_x", "world_y"]


def write_parquet(df: pd.DataFrame, path: str | Path, meta: dict) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({k: str(v) for k, v in meta.items()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_parquet_with_meta(path: str | Path) -> tuple[pd.DataFrame, dict]:
    table = pq.read_table(path)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    return table.to_pandas(), meta


def frame_range(video: str | Path, start: int, end: int | None) -> tuple[int, int]:
    n = probe_video_stream(video)["nb_frames"]
    return start, min(end if end is not None else n, n)


def detect_range(video: str | Path, start: int, end: int, det_cfg: dict, roi, out: str | Path) -> pd.DataFrame:
    """Run the detector on frames [start, end) and cache every detection (no filtering, no IDs)."""
    timestamps = build_pts_index(video)
    detector = Detector(det_cfg, roi=roi)
    rows = []
    t0 = time.perf_counter()
    first = True
    for idx, frame in iter_frames(video, start, end):
        if first:  # time only the detection loop, not the skip to `start`
            t0, first = time.perf_counter(), False
        det = detector(frame)
        for (x1, y1, x2, y2), c, k in zip(det.xyxy, det.confidence, det.class_id):
            rows.append((idx, float(timestamps[idx]), int(k), detector.names[int(k)], float(c),
                         float(x1), float(y1), float(x2), float(y2)))
        n = idx - start + 1
        if n % 100 == 0:
            print(f"  detect {n}/{end - start}, {(time.perf_counter() - t0) / n:.3f} s/frame", flush=True)
    df = pd.DataFrame(rows, columns=DETECTION_COLS)
    write_parquet(df, out, {
        "kind": "detections", "source_video": Path(video).name, "start_frame": start, "end_frame": end,
        "avg_fps": average_fps(video), "model": det_cfg["model"], "imgsz": det_cfg["imgsz"], "conf": det_cfg["conf"],
    })
    print(f"detect: {len(df)} detections over {end - start} frames -> {out}")
    return df


def track_detections(dets: pd.DataFrame, start: int, end: int, fps: float, tracker_cfg: dict) -> pd.DataFrame:
    """ByteTrack over all cached detections. The road mask is applied afterwards, per track."""
    tracker = make_tracker(tracker_cfg, fps)
    by_frame = {k: g for k, g in dets.groupby("frame_idx")}
    names = dict(zip(dets.class_id, dets.class_name))
    rows = []
    for idx in range(start, end):  # every frame must update the tracker, including empty ones
        g = by_frame.get(idx)
        if g is None:
            det = sv.Detections.empty()
            ts = None
        else:
            det = sv.Detections(xyxy=g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32),
                                confidence=g.conf.to_numpy(dtype=np.float32),
                                class_id=g.class_id.to_numpy(dtype=int))
            ts = float(g.timestamp_s.iloc[0])
        det = tracker.update_with_detections(det)
        det = det[det.tracker_id != -1]
        if len(det) == 0:
            continue
        for (x1, y1, x2, y2), c, k, t in zip(det.xyxy, det.confidence, det.class_id, det.tracker_id):
            rows.append((idx, ts, int(t), int(k), names[int(k)], float(c), float(x1), float(y1), float(x2), float(y2),
                         float((x1 + x2) / 2), float(y2), np.nan, np.nan))
    return pd.DataFrame(rows, columns=TRAJECTORY_COLS)


def keep_road_tracks(traj: pd.DataFrame, polygons: list[np.ndarray], min_fraction: float) -> tuple[pd.DataFrame, int]:
    """Keep tracks whose bottom-centre is on the road for at least min_fraction of their frames.

    Per-track, not per-detection, so a car that touches the polygon edge keeps one continuous track.
    Returns the kept rows and the number of dropped tracks.
    """
    if not polygons or traj.empty:
        return traj, 0
    on_road = points_inside(traj[["anchor_x", "anchor_y"]].to_numpy(), polygons)
    fraction = pd.Series(on_road, index=traj.index).groupby(traj.track_id).mean()
    keep = fraction[fraction >= min_fraction].index
    return traj[traj.track_id.isin(keep)].reset_index(drop=True), int(len(fraction) - len(keep))


def smooth_range(traj: pd.DataFrame, video: str | Path, smoothing_cfg: dict) -> tuple[pd.DataFrame, dict]:
    return smooth_trajectories(traj, build_pts_index(video), smoothing_cfg)
