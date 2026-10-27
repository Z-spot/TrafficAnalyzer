"""Track quality numbers, used to compare raw and smoothed trajectories."""

import numpy as np
import pandas as pd


def track_stats(df: pd.DataFrame, short_frames: int = 30) -> dict:
    df = df.sort_values(["track_id", "frame_idx"])
    g = df.groupby("track_id")
    n = g.size()
    span = g.frame_idx.max() - g.frame_idx.min() + 1
    ncls = g.class_name.nunique()
    # jitter: mean |second difference| of box centre and size, independent of steady motion
    cx, cy = (df.x1 + df.x2) / 2, (df.y1 + df.y2) / 2
    feats = {"cx": cx, "cy": cy, "w": df.x2 - df.x1, "h": df.y2 - df.y1}
    jitter = {}
    for name, s in feats.items():
        d2 = s.groupby(df.track_id).diff().groupby(df.track_id).diff().abs()
        consecutive = df.frame_idx.groupby(df.track_id).diff(2) == 2
        jitter[f"jitter_{name}_px"] = round(float(d2[consecutive].mean()), 3)
    return {
        "rows": len(df),
        "tracks": int(len(n)),
        f"tracks_under_{short_frames}_frames": int((n < short_frames).sum()),
        "tracks_with_gaps": int((span > n).sum()),
        "missing_rows_in_gaps": int((span - n).sum()),
        "tracks_with_class_flips": int((ncls > 1).sum()),
        "interpolated_rows": int(df.conf.isna().sum()),
        **jitter,
    }
