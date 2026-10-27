"""Video reading helpers. Frame order is presentation order, frame_idx starts at 0."""

import json
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np


def probe_video_stream(path: str | Path) -> dict:
    """ffprobe metadata of the first video stream (needs ffprobe on PATH)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,nb_frames,r_frame_rate,avg_frame_rate,time_base,start_pts,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    s["width"], s["height"], s["nb_frames"] = int(s["width"]), int(s["height"]), int(s["nb_frames"])
    return s


def open_video(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"OpenCV could not open {path}")
    return cap


def count_decoded_frames(path: str | Path) -> int:
    """Count frames by decoding the whole stream. Slow, but exact, unlike CAP_PROP_FRAME_COUNT."""
    cap = open_video(path)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def read_frames(path: str | Path, frame_indices: Iterable[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, BGR frame) for the requested indices by reading sequentially from frame 0.

    Sequential reading avoids OpenCV seeking, which is not frame-accurate on long-GOP HEVC.
    """
    wanted = sorted(set(frame_indices))
    cap = open_video(path)
    idx = 0
    try:
        for target in wanted:
            while idx < target:
                if not cap.grab():
                    raise EOFError(f"{path} ended at frame {idx}, before frame {target}")
                idx += 1
            ok, frame = cap.read()
            if not ok:
                raise EOFError(f"{path} ended at frame {idx}")
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def build_pts_index(path: str | Path) -> np.ndarray:
    """Presentation timestamps in seconds, one per frame, in presentation order, from the container.

    Reads packet timestamps with ffprobe (no decoding). HEVC with B-frames stores packets in decode
    order, so the timestamps are sorted. Index i is the timestamp of frame_idx i. Time zero is the
    first frame. Raises if any gap is not roughly one frame interval (dropped or duplicated frames).
    """
    stream = probe_video_stream(path)
    num, den = (int(x) for x in stream["time_base"].split("/"))
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    pts = np.sort(np.array([int(x) for x in out.split()], dtype=np.int64))
    if len(pts) != stream["nb_frames"]:
        raise ValueError(f"{len(pts)} packets but ffprobe nb_frames={stream['nb_frames']}")
    gaps = np.diff(pts)
    if gaps.min() <= 0 or gaps.max() > 1.5 * np.median(gaps):
        raise ValueError(f"unexpected frame gaps in ticks: min={gaps.min()} max={gaps.max()} median={np.median(gaps)}")
    return (pts - pts[0]) * num / den


def average_fps(path: str | Path) -> float:
    """Average frame rate from the container (avg_frame_rate), not an assumed constant."""
    n, d = (int(x) for x in probe_video_stream(path)["avg_frame_rate"].split("/"))
    return n / d


def iter_frames(path: str | Path, start_frame: int, end_frame: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, BGR frame) for start_frame <= idx < end_frame, in original-video indices.

    Reads sequentially from frame 0 (skipped frames are only grabbed, not converted), because
    seeking is not frame-accurate on long-GOP HEVC.
    """
    cap = open_video(path)
    try:
        for _ in range(start_frame):
            if not cap.grab():
                raise EOFError(f"{path} has fewer than {start_frame} frames")
        for idx in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                return
            yield idx, frame
    finally:
        cap.release()
