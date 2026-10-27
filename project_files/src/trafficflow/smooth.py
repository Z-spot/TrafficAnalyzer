"""Offline clean-up of tracks. Uses past and future frames, so it is only valid for recorded video.

Input and output are trajectory tables in the CLAUDE.md schema.
"""

import numpy as np
import pandas as pd

# Interpolated rows have no detector confidence.
INTERPOLATED_CONF = np.nan


def vote_class(df: pd.DataFrame) -> pd.DataFrame:
    """One class per track: the class with the highest summed confidence."""
    df = df.copy()
    weight = df.groupby(["track_id", "class_id", "class_name"], sort=False).conf.sum().reset_index()
    best = weight.sort_values("conf", ascending=False).drop_duplicates("track_id").set_index("track_id")
    df["class_id"] = df.track_id.map(best.class_id)
    df["class_name"] = df.track_id.map(best.class_name)
    return df


def _track_table(df: pd.DataFrame, k: int) -> list[dict]:
    """First/last frame, end positions and end velocities (px per frame, from k rows) of every track."""
    rows = []
    for tid, t in df.sort_values("frame_idx").groupby("track_id", sort=False):
        f, ax, ay = t.frame_idx.to_numpy(), t.anchor_x.to_numpy(), t.anchor_y.to_numpy()

        def vel(sl: slice) -> np.ndarray:
            ff, xx, yy = f[sl], ax[sl], ay[sl]
            if len(ff) < 3 or ff[-1] == ff[0]:
                return np.zeros(2)
            return np.array([xx[-1] - xx[0], yy[-1] - yy[0]]) / (ff[-1] - ff[0])

        rows.append(dict(id=tid, first=int(f[0]), last=int(f[-1]), cls=int(t.class_id.iloc[0]),
                         p0=np.array([ax[0], ay[0]]), p1=np.array([ax[-1], ay[-1]]),
                         v_end=vel(slice(-k, None)), v_start=vel(slice(0, k))))
    return rows


def _link_candidates(rows: list[dict], cfg: dict) -> dict[tuple, float]:
    """Pairs (A, B): B starts after A ends, same class, close to where A's motion predicts (and vice versa)."""
    cands = {}
    for a in rows:
        for b in rows:
            gap = b["first"] - a["last"]
            if a["id"] == b["id"] or a["cls"] != b["cls"] or gap < 1 or gap > cfg["stitch_max_gap_frames"]:
                continue
            err_fwd = np.linalg.norm(b["p0"] - (a["p1"] + a["v_end"] * gap))    # where A should be when B starts
            err_bwd = np.linalg.norm(a["p1"] - (b["p0"] - b["v_start"] * gap))  # where B should have been when A ended
            err = (err_fwd + err_bwd) / 2
            if err <= cfg["stitch_base_tol_px"] + cfg["stitch_tol_per_frame_px"] * gap:
                cands[(a["id"], b["id"])] = float(err)
    return cands


def stitch_tracks(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[tuple]]:
    """Merge tracks that are very likely one vehicle whose track was lost and restarted with a new ID.

    A link is accepted only if both sides choose each other. For a track A, the candidates are grouped by
    time: later pieces of the same vehicle (which start after the first candidate ends) are not rivals, only
    candidates that exist at the same time are. Within a group the best match wins unless a rival is nearly
    as good (within stitch_ambiguity_ratio times the best error, and never closer than stitch_ambiguity_min_px),
    which stops queued cars side by side being swapped. Repeats a few rounds so a vehicle broken into three
    pieces is rejoined. The earlier track keeps its ID.
    Returns the table and the accepted links as (id_a, id_b, last_frame_a, first_frame_b, error_px).
    """
    df = df.copy()
    all_links: list[tuple] = []
    for _ in range(4):
        rows = _track_table(df, cfg["stitch_velocity_frames"])
        info = {r["id"]: r for r in rows}
        cands = _link_candidates(rows, cfg)
        if not cands:
            break
        by_a: dict = {}
        by_b: dict = {}
        for (a, b), err in cands.items():
            by_a.setdefault(a, []).append((err, b))
            by_b.setdefault(b, []).append((err, a))

        def choose(lst: list[tuple], earliest_first: bool):
            """Pick the partner for one track, or None if the best match has a close rival at the same time."""
            if earliest_first:   # A's successors: the earliest-starting group is the next piece
                lst = sorted(lst, key=lambda e: info[e[1]]["first"])
                group = [e for e in lst if info[e[1]]["first"] <= info[lst[0][1]]["last"]]
            else:                # B's predecessors: the latest-ending group is the previous piece
                lst = sorted(lst, key=lambda e: -info[e[1]]["last"])
                group = [e for e in lst if info[e[1]]["last"] >= info[lst[0][1]]["first"]]
            group.sort()
            if len(group) > 1 and group[1][0] < max(cfg["stitch_ambiguity_ratio"] * group[0][0],
                                                     cfg["stitch_ambiguity_min_px"]):
                return None
            return group[0][1]

        choice_a = {a: choose(lst, True) for a, lst in by_a.items()}
        choice_b = {b: choose(lst, False) for b, lst in by_b.items()}
        accepted = {a: b for a, b in choice_a.items() if b is not None and choice_b.get(b) == a}
        if not accepted:
            break
        for a, b in accepted.items():
            all_links.append((a, b, info[a]["last"], info[b]["first"], round(cands[(a, b)], 1)))

        def root(x):
            seen = set()
            while x in accepted_rev and x not in seen:
                seen.add(x)
                x = accepted_rev[x]
            return x

        accepted_rev = {b: a for a, b in accepted.items()}
        df["track_id"] = df.track_id.map(lambda t: root(t))
    return df, all_links


def fill_gaps(df: pd.DataFrame, timestamps: np.ndarray, max_gap: int) -> pd.DataFrame:
    """Linearly interpolate boxes over holes of at most max_gap frames inside a track."""
    out = []
    for tid, t in df.groupby("track_id", sort=False):
        t = t.sort_values("frame_idx")
        full = pd.RangeIndex(t.frame_idx.min(), t.frame_idx.max() + 1, name="frame_idx")
        t = t.set_index("frame_idx").reindex(full)
        missing = t.track_id.isna()
        # label each run of missing rows with its length, keep only runs short enough to fill
        run_id = (missing != missing.shift()).cumsum()
        run_len = missing.groupby(run_id).transform("size")
        fillable = missing & (run_len <= max_gap)
        cols = ["x1", "y1", "x2", "y2"]
        t[cols] = t[cols].interpolate(method="linear", limit_area="inside")
        keep = ~missing | fillable
        t = t[keep].copy()
        filled = fillable[keep]
        t["track_id"] = tid
        t["class_id"] = t.class_id.ffill().bfill()
        t["class_name"] = t.class_name.ffill().bfill()
        t.loc[filled, "conf"] = INTERPOLATED_CONF
        t = t.reset_index()
        t["timestamp_s"] = timestamps[t.frame_idx.to_numpy()]
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else df.copy()


def drop_short_tracks(df: pd.DataFrame, min_frames: int) -> pd.DataFrame:
    n = df.groupby("track_id").frame_idx.transform("size")
    return df[n >= min_frames].copy()


def smooth_boxes(df: pd.DataFrame, box_window: int, size_window: int) -> pd.DataFrame:
    """Centred rolling mean of box centre, width and height per track; rebuilds x1..y2 and the anchor."""
    df = df.sort_values(["track_id", "frame_idx"]).copy()
    cx, cy = (df.x1 + df.x2) / 2, (df.y1 + df.y2) / 2
    w, h = df.x2 - df.x1, df.y2 - df.y1
    tid = df.track_id

    def roll(s: pd.Series, win: int) -> pd.Series:
        return s.groupby(tid).transform(lambda x: x.rolling(win, center=True, min_periods=1).mean())

    cx, cy = roll(cx, box_window), roll(cy, box_window)
    w, h = roll(w, size_window), roll(h, size_window)
    df["x1"], df["x2"] = cx - w / 2, cx + w / 2
    df["y1"], df["y2"] = cy - h / 2, cy + h / 2
    df["anchor_x"], df["anchor_y"] = cx, df.y2
    return df


def smooth_trajectories(df: pd.DataFrame, timestamps: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Full clean-up. Returns the smoothed table and a small report (for the rejection rate)."""
    n_tracks_in = df.track_id.nunique()
    df = vote_class(df)
    df, links = stitch_tracks(df, cfg)
    df = vote_class(df)
    # holes inside a track, original or stitched, are filled up to the stitching limit
    df = fill_gaps(df, timestamps, max(cfg["max_gap_frames"], cfg["stitch_max_gap_frames"]))
    df = drop_short_tracks(df, cfg["min_track_frames"])
    df = smooth_boxes(df, cfg["box_window"], cfg["size_window"])
    df = df.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)
    report = {"tracks_in": n_tracks_in, "tracks_out": df.track_id.nunique(),
              "interpolated_rows": int(df.conf.isna().sum()), "stitched_links": len(links), "links": links}
    return df, report
