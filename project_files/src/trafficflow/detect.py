"""Vehicle detection with ultralytics YOLO. Boxes are always returned in full-frame pixels."""

import numpy as np
import supervision as sv
from ultralytics import YOLO


class Detector:
    def __init__(self, cfg: dict, roi: list[int] | None = None):
        self.cfg = cfg
        self.roi = roi
        self.model = YOLO(cfg["model"])
        self.names: dict[int, str] = self.model.names
        missing = [c for c in cfg["vehicle_classes"] if c not in self.names]
        if missing:
            raise ValueError(f"vehicle_classes {missing} not in model.names")

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        x0, y0 = 0, 0
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi
            frame = frame[y0:y1, x0:x1]
        res = self.model.predict(
            frame, imgsz=self.cfg["imgsz"], conf=self.cfg["conf"], iou=self.cfg["iou"],
            classes=self.cfg["vehicle_classes"], device=self.cfg["device"], verbose=False,
        )[0]
        det = sv.Detections.from_ultralytics(res)
        det.xyxy = det.xyxy + np.array([x0, y0, x0, y0], dtype=det.xyxy.dtype)
        return det
