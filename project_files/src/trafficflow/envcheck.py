"""Environment and compatibility checks (Phase 1, Task 1). Each check returns a CheckResult."""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from trafficflow.video import count_decoded_frames, open_video, read_frames

# Standard COCO names the configured vehicle class IDs are expected to map to.
VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_mps() -> CheckResult:
    import torch

    built, available = torch.backends.mps.is_built(), torch.backends.mps.is_available()
    if not (built and available):
        return CheckResult("mps", False, f"torch {torch.__version__}: is_built={built} is_available={available}")
    x = torch.arange(6, dtype=torch.float32, device="mps").reshape(2, 3)
    y = (x @ x.T).cpu()
    ok = torch.equal(y, torch.tensor([[5.0, 14.0], [14.0, 50.0]]))
    return CheckResult("mps", ok, f"torch {torch.__version__}: is_built={built} is_available={available}, matmul on mps ok={ok}")


def check_model_classes(model, vehicle_class_ids: list[int]) -> CheckResult:
    mapping = {i: model.names[i] for i in vehicle_class_ids}
    ok = set(mapping.values()) == VEHICLE_NAMES
    return CheckResult("model classes", ok, f"config ids -> model.names: {mapping}")


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy boxes a (N,4) and b (M,4)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def check_mps_matches_cpu(model, image: np.ndarray, det_cfg: dict, min_iou: float = 0.95) -> CheckResult:
    boxes, times = {}, {}
    for device in ("cpu", "mps"):
        kwargs = dict(imgsz=det_cfg["imgsz"], conf=det_cfg["conf"], iou=det_cfg["iou"],
                      classes=det_cfg["vehicle_classes"], device=device, verbose=False)
        model.predict(image, **kwargs)  # warm-up
        t0 = time.perf_counter()
        res = model.predict(image, **kwargs)[0]
        times[device] = time.perf_counter() - t0
        boxes[device] = res.boxes.xyxy.cpu().numpy()
    cpu, mps = boxes["cpu"], boxes["mps"]
    detail = f"boxes cpu={len(cpu)} mps={len(mps)}, one frame cpu={times['cpu']:.2f}s mps={times['mps']:.2f}s"
    if len(cpu) == 0 or len(mps) == 0:
        return CheckResult("mps vs cpu", False, detail + " (no boxes to compare)")
    best = box_iou(cpu, mps).max(axis=1)  # best MPS match for each CPU box
    n_low = int((best < min_iou).sum())
    detail += f", matched IoU min={best.min():.3f} median={np.median(best):.3f}, below {min_iou}: {n_low}"
    return CheckResult("mps vs cpu", len(cpu) == len(mps) and n_low == 0, detail)


def check_opencv_open(video: Path, expected_wh: tuple[int, int]) -> CheckResult:
    cap = open_video(video)
    backend = cap.getBackendName()
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    ok = backend == "FFMPEG" and (w, h) == expected_wh
    return CheckResult("opencv open", ok, f"backend={backend} size={w}x{h} metadata frame_count={meta_count} fps={meta_fps:.4f}")


def check_decoded_frame_count(video: Path, expected: int) -> CheckResult:
    t0 = time.perf_counter()
    n = count_decoded_frames(video)
    dt = time.perf_counter() - t0
    return CheckResult("decoded frame count", n == expected,
                       f"decoded {n} frames (expected {expected}) in {dt:.0f}s, {n / dt:.1f} frames/s decode-only")


def check_reference_frames(video: Path, refs: dict[int, Path], search: int = 3) -> CheckResult:
    """Compare decoded frames idx-search..idx+search with each reference PNG.

    Passes if, for every reference, the closest decoded frame is exactly idx.
    Mean absolute difference is reported because YUV->RGB conversion may differ by a level or so
    between OpenCV and whatever tool extracted the PNGs.
    """
    images = {idx: cv2.imread(str(p), cv2.IMREAD_COLOR) for idx, p in refs.items()}
    wanted = {i for idx in refs for i in range(max(0, idx - search), idx + search + 1)}
    mad: dict[int, dict[int, float]] = {idx: {} for idx in refs}
    for i, frame in read_frames(video, wanted):
        for idx, ref in images.items():
            if abs(i - idx) <= search:
                mad[idx][i] = float(np.abs(frame.astype(np.int16) - ref.astype(np.int16)).mean())
    lines, ok = [], True
    for idx in sorted(refs):
        best = min(mad[idx], key=mad[idx].get)
        exact = mad[idx][idx] == 0.0
        ok &= best == idx
        others = min(v for k, v in mad[idx].items() if k != idx)
        lines.append(f"frame {idx}: best match {best}, MAD at idx={mad[idx][idx]:.3f} "
                     f"(next best neighbour {others:.3f}), pixel-identical={exact}")
    return CheckResult("reference frames", ok, "; ".join(lines))
