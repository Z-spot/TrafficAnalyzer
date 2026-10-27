import numpy as np
import pandas as pd

from trafficflow.pipeline import TRAJECTORY_COLS, keep_road_tracks

SQUARE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)


def track(tid, xs):
    rows = [(f, f / 30, tid, 2, "car", 0.9, x - 5, 80, x + 5, 90, x, 90, np.nan, np.nan) for f, x in enumerate(xs)]
    return pd.DataFrame(rows, columns=TRAJECTORY_COLS)


def test_on_road_parked_and_half_in_tracks():
    on_road = track(1, [10] * 20)
    parked = track(2, [500] * 20)
    mostly_in = track(3, [50] * 12 + [150] * 8)    # 60% inside, leaves the polygon at the end
    mostly_out = track(4, [50] * 8 + [150] * 12)   # 40% inside
    df = pd.concat([on_road, parked, mostly_in, mostly_out], ignore_index=True)
    kept, dropped = keep_road_tracks(df, [SQUARE], 0.5)
    assert sorted(kept.track_id.unique()) == [1, 3] and dropped == 2
    assert len(kept[kept.track_id == 3]) == 20  # the part outside the polygon is kept, one continuous track


def test_no_polygons_keeps_everything():
    df = track(1, [500] * 5)
    kept, dropped = keep_road_tracks(df, [], 0.5)
    assert len(kept) == 5 and dropped == 0
