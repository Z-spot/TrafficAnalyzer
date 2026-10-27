import numpy as np
import pandas as pd

from trafficflow.smooth import drop_short_tracks, fill_gaps, smooth_boxes, smooth_trajectories, stitch_tracks, vote_class
from trafficflow.stats import track_stats

TS = np.arange(400) / 30.0
CFG = dict(max_gap_frames=10, min_track_frames=15, box_window=7, size_window=15, stitch_max_gap_frames=60,
           stitch_base_tol_px=60, stitch_tol_per_frame_px=3, stitch_ambiguity_ratio=1.5, stitch_ambiguity_min_px=25, stitch_velocity_frames=8)


def make_track(tid, frames, x0=100.0, vx=2.0, cls=(2, "car"), conf=0.8):
    rows = []
    for f in frames:
        x = x0 + vx * f
        rows.append(dict(frame_idx=f, timestamp_s=TS[f], track_id=tid, class_id=cls[0], class_name=cls[1], conf=conf,
                         x1=x, y1=50.0, x2=x + 40, y2=90.0, anchor_x=x + 20, anchor_y=90.0, world_x=np.nan, world_y=np.nan))
    return pd.DataFrame(rows)


def test_vote_class_uses_confidence_weighted_majority():
    t = make_track(1, range(10))
    t.loc[:2, ["class_id", "class_name"]] = [7, "truck"]
    t.loc[:2, "conf"] = 0.3
    out = vote_class(t)
    assert set(out.class_name) == {"car"}


def test_fill_gaps_interpolates_short_hole_and_marks_conf_nan():
    t = make_track(1, [0, 1, 2, 6, 7])  # hole of 3 frames (3, 4, 5)
    out = fill_gaps(t, TS, max_gap=5).sort_values("frame_idx")
    assert out.frame_idx.tolist() == list(range(8))
    filled = out[out.frame_idx.isin([3, 4, 5])]
    assert filled.conf.isna().all() and out[~out.frame_idx.isin([3, 4, 5])].conf.notna().all()
    np.testing.assert_allclose(out.x1.to_numpy(), 100 + 2 * np.arange(8))  # linear motion stays linear
    np.testing.assert_allclose(out.timestamp_s.to_numpy(), TS[:8])


def test_fill_gaps_leaves_long_hole_open():
    t = make_track(1, [0, 1, 20, 21])
    out = fill_gaps(t, TS, max_gap=5)
    assert sorted(out.frame_idx) == [0, 1, 20, 21]


def test_drop_short_tracks():
    df = pd.concat([make_track(1, range(20)), make_track(2, range(5))])
    assert drop_short_tracks(df, 15).track_id.unique().tolist() == [1]


def test_smoothing_keeps_straight_motion_and_reduces_jitter():
    t = make_track(1, range(60))
    rng = np.random.default_rng(0)
    noisy = t.copy()
    for c in ("x1", "x2", "y1", "y2"):
        noisy[c] = noisy[c] + rng.normal(0, 1.5, len(noisy))
    out = smooth_boxes(noisy, box_window=7, size_window=15)
    before, after = track_stats(noisy), track_stats(out)
    assert after["jitter_cx_px"] < before["jitter_cx_px"] / 2
    assert after["jitter_h_px"] < before["jitter_h_px"] / 2
    # centre of the clean track is recovered inside the window interior
    mid = out[(out.frame_idx > 10) & (out.frame_idx < 50)]
    np.testing.assert_allclose((mid.x1 + mid.x2) / 2, 120 + 2 * mid.frame_idx, atol=1.5)
    np.testing.assert_allclose(mid.anchor_y, mid.y2)


def test_full_pipeline_report():
    df = pd.concat([make_track(1, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                    make_track(2, range(4))])
    cfg = CFG
    out, report = smooth_trajectories(df, TS, cfg)
    assert report["tracks_in"] == 2 and report["tracks_out"] == 1 and report["interpolated_rows"] == 2
    assert out.frame_idx.tolist() == list(range(20))


def test_stitch_links_a_lost_and_restarted_track_and_pipeline_fills_the_gap():
    a = make_track(1, range(0, 40), x0=100, vx=3)         # ends at frame 39
    b = make_track(2, range(55, 100), x0=100, vx=3)       # same straight motion, restarts 16 frames later
    out, links = stitch_tracks(pd.concat([a, b]), CFG)
    assert out.track_id.unique().tolist() == [1] and [(l[0], l[1]) for l in links] == [(1, 2)]
    full, report = smooth_trajectories(pd.concat([a, b]), TS, CFG)
    assert report["stitched_links"] == 1 and report["tracks_out"] == 1
    assert full.frame_idx.tolist() == list(range(100))    # the hidden stretch is interpolated, no hole left


def test_stitch_does_not_link_different_classes_or_overlapping_tracks():
    a = make_track(1, range(0, 40), vx=3)
    truck = make_track(2, range(55, 100), x0=100, vx=3, cls=(7, "truck"))
    assert stitch_tracks(pd.concat([a, truck]), CFG)[1] == []
    overlap = make_track(3, range(30, 80), x0=100, vx=3)
    assert stitch_tracks(pd.concat([a, overlap]), CFG)[1] == []


def test_stitch_skips_ambiguous_candidates():
    a = make_track(1, range(0, 40), x0=100, vx=0)
    b1 = make_track(2, range(50, 100), x0=100, vx=0)
    b2 = make_track(3, range(50, 100), x0=112, vx=0)       # nearly as good a match as b1
    assert stitch_tracks(pd.concat([a, b1, b2]), CFG)[1] == []


def test_stitch_does_not_join_distant_stationary_queue_cars():
    a = make_track(1, range(0, 40), x0=100, vx=0)
    b = make_track(2, range(50, 100), x0=400, vx=0)        # 300 px away, tolerance is 60 + 3*10
    assert stitch_tracks(pd.concat([a, b]), CFG)[1] == []


def test_stitch_rejoins_a_vehicle_broken_into_three_pieces():
    parts = [make_track(1, range(0, 30), vx=3), make_track(2, range(40, 70), vx=3), make_track(3, range(80, 110), vx=3)]
    out, links = stitch_tracks(pd.concat(parts), CFG)
    assert out.track_id.nunique() == 1 and len(links) == 2
