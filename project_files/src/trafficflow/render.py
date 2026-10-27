"""Debug video from a trajectory parquet plus the original video. No detection is run."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from trafficflow.video import average_fps, iter_frames, probe_video_stream


def render_debug_video(video: str | Path, traj: pd.DataFrame, start: int, end: int, out: str | Path,
                       height: int = 1080, polygons: list[np.ndarray] | None = None, banner: str = "") -> None:
    probe = probe_video_stream(video)
    scale = height / probe["height"]
    size = (round(probe["width"] * scale), height)
    box_ann = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK, thickness=2)
    label_ann = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK, text_scale=0.5, text_padding=4)
    by_frame = {k: g for k, g in traj.groupby("frame_idx")}
    polys_px = [(p * scale).astype(np.int32).reshape(-1, 1, 2) for p in (polygons or [])]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with sv.VideoSink(str(out), sv.VideoInfo(size[0], size[1], average_fps(video))) as sink:
        for idx, frame in iter_frames(video, start, end):
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            if polys_px:
                cv2.polylines(small, polys_px, True, (0, 255, 255), 2)
            g = by_frame.get(idx)
            if g is not None:
                det = sv.Detections(xyxy=g[["x1", "y1", "x2", "y2"]].to_numpy() * scale,
                                    class_id=g.class_id.to_numpy(dtype=int), tracker_id=g.track_id.to_numpy(dtype=int))
                labels = [f"{n} #{t}" for n, t in zip(g.class_name, g.track_id)]
                small = box_ann.annotate(small, det)
                small = label_ann.annotate(small, det, labels)
            if banner:
                cv2.putText(small, f"{banner}  frame {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(small, f"{banner}  frame {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
            sink.write_frame(small)
    print(f"render: {out}")
