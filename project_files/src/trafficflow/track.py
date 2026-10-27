"""ByteTrack wrapper. Frame rate always comes from the video file."""

import warnings

import supervision as sv


def make_tracker(tracker_cfg: dict, fps: float):
    if tracker_cfg["type"] != "bytetrack":
        raise ValueError(f"unsupported tracker {tracker_cfg['type']!r}")
    with warnings.catch_warnings():
        # sv.ByteTrack is deprecated in supervision 0.30.9 but still present. See docs/decisions.md.
        warnings.simplefilter("ignore", FutureWarning)
        return sv.ByteTrack(
            track_activation_threshold=tracker_cfg["track_activation_threshold"],
            lost_track_buffer=tracker_cfg["lost_track_buffer"],
            minimum_matching_threshold=tracker_cfg["minimum_matching_threshold"],
            minimum_consecutive_frames=tracker_cfg["minimum_consecutive_frames"],
            frame_rate=fps,
        )
