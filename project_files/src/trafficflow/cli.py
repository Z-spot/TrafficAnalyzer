"""Commands. Stages: detect (slow, cached) -> track (road mask + ByteTrack) -> smooth -> render.

  python -m trafficflow.cli run --name NAME --start-frame N --end-frame M     all four stages
  python -m trafficflow.cli detect|track|smooth|render ...                    one stage, from cached files
"""

import argparse
from pathlib import Path

from trafficflow.config import PROJECT_ROOT, load_config
from trafficflow.pipeline import (
    detect_range,
    frame_range,
    keep_road_tracks,
    read_parquet_with_meta,
    smooth_range,
    track_detections,
    write_parquet,
)
from trafficflow.render import render_debug_video
from trafficflow.roadmask import load_road_polygons
from trafficflow.video import average_fps

DEFAULT_VIDEO = str(PROJECT_ROOT / "data/raw/main_video.MOV")
INTERIM = PROJECT_ROOT / "data/interim"


def paths(name: str) -> dict[str, Path]:
    return {
        "dets": INTERIM / "detections" / f"{name}.parquet",
        "raw": INTERIM / f"{name}.raw.parquet",
        "smooth": INTERIM / f"{name}.smooth.parquet",
        "raw_video": INTERIM / "debug" / f"{name}_raw.mp4",
        "smooth_video": INTERIM / "debug" / f"{name}_smooth.mp4",
    }


def do_detect(video, start, end, out):
    cfg = load_config()
    start, end = frame_range(video, start, end)
    detect_range(video, start, end, cfg["detector"], cfg["video"]["roi"], out)


def do_track(video, dets_path, out):
    cfg = load_config()
    dets, meta = read_parquet_with_meta(dets_path)
    start, end, fps = int(meta["start_frame"]), int(meta["end_frame"]), float(meta["avg_fps"])
    polygons = load_road_polygons()
    traj = track_detections(dets, start, end, fps, cfg["tracker"])
    n_all = traj.track_id.nunique()
    traj, n_dropped = keep_road_tracks(traj, polygons, cfg["tracker"]["min_inside_fraction"])
    print(f"track: road mask {'on, ' + str(len(polygons)) + ' polygon(s)' if polygons else 'OFF (none in config/zones.yaml)'}; "
          f"{n_dropped} of {n_all} tracks dropped as off-road")
    write_parquet(traj, out, {"kind": "trajectories", "smoothed": False, "source_video": Path(video).name,
                              "start_frame": start, "end_frame": end, "avg_fps": fps,
                              "road_polygons": len(polygons), "off_road_tracks_dropped": n_dropped,
                              **{f"det_{k}": v for k, v in meta.items() if k in ("model", "imgsz", "conf")}})
    print(f"track: {len(traj)} rows, {traj.track_id.nunique()} tracks -> {out}")


def do_smooth(video, raw_path, out):
    cfg = load_config()
    traj, meta = read_parquet_with_meta(raw_path)
    smoothed, report = smooth_range(traj, video, cfg["smoothing"])
    links = report.pop("links")
    meta.update({"smoothed": True, **{f"smooth_{k}": v for k, v in report.items()}})
    write_parquet(smoothed, out, meta)
    print(f"smooth: {report} -> {out}")
    for a, b, end_a, start_b, err in links:
        print(f"  stitched track {b} into {a}: ended frame {end_a}, restarted frame {start_b}, {err} px off")


def do_render(video, traj_path, out, height, banner=""):
    traj, meta = read_parquet_with_meta(traj_path)
    render_debug_video(video, traj, int(meta["start_frame"]), int(meta["end_frame"]), out, height,
                       load_road_polygons(), banner)


def main() -> None:
    ap = argparse.ArgumentParser(prog="trafficflow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "detect", "track", "smooth", "render"):
        p = sub.add_parser(name)
        p.add_argument("--video", default=DEFAULT_VIDEO)
        p.add_argument("--name", default=None, help="base name for the files under data/interim/")
        p.add_argument("--start-frame", type=int, default=0)
        p.add_argument("--end-frame", type=int, default=None, help="exclusive; default is end of video")
        p.add_argument("--debug-height", type=int, default=1080)
        p.add_argument("--in", dest="inp", default=None)
        p.add_argument("--out", default=None)
    args = ap.parse_args()
    p = paths(args.name) if args.name else {}

    if args.cmd == "run":
        do_detect(args.video, args.start_frame, args.end_frame, p["dets"])
        do_track(args.video, p["dets"], p["raw"])
        do_smooth(args.video, p["raw"], p["smooth"])
        do_render(args.video, p["raw"], p["raw_video"], args.debug_height, "RAW")
        do_render(args.video, p["smooth"], p["smooth_video"], args.debug_height, "SMOOTHED")
    elif args.cmd == "detect":
        do_detect(args.video, args.start_frame, args.end_frame, args.out or p["dets"])
    elif args.cmd == "track":
        do_track(args.video, args.inp or p["dets"], args.out or p["raw"])
    elif args.cmd == "smooth":
        do_smooth(args.video, args.inp or p["raw"], args.out or p["smooth"])
    elif args.cmd == "render":
        do_render(args.video, args.inp or p["smooth"], args.out or p["smooth_video"], args.debug_height)


if __name__ == "__main__":
    main()
