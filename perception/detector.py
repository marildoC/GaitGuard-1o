from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

try:
    import torch
except Exception:
    torch = None

from ultralytics import YOLO

from schemas import Frame

logger = logging.getLogger(__name__)



@dataclass
class Detection:
    """
    Single detection produced by the detector.

    Coordinates are in absolute pixel space in the original frame:
        (x1, y1) = top-left corner
        (x2, y2) = bottom-right corner
    """
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


@dataclass
class DetectorConfig:
    """
    Configuration for the YOLO detector.

    - model_path: path or model name for Ultralytics YOLO
    - imgsz: inference resolution (model will internally letterbox to this)
    - conf: confidence threshold
    - iou: NMS IoU threshold
    - device: "cuda", "cpu", "mps", "0", "0,1", etc.  If None → auto.
    - half: use FP16 on CUDA for speed (ignored if no CUDA)
    - classes: which numeric class IDs to keep. By default keep only 'person' (0).
    """
    model_path: str = "yolo11n.pt"  # or "yolov8n.pt" depending on what you installed
    imgsz: int = 640
    conf: float = 0.35
    iou: float = 0.45
    device: Optional[str] = None
    half: bool = True
    classes: Optional[Sequence[int]] = (0,)



class Detector:
    """
    Thin wrapper around an Ultralytics YOLO model.

    Responsibilities:
      - Load YOLO model once.
      - Run inference on a Frame (or raw np.ndarray).
      - Return clean Detection objects for people only (by default).
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self._model: Optional[YOLO] = None
        self._device_str: str = "cpu"

        self._load_model()

    def _select_device(self) -> str:
        """
        Decide which device string to use for YOLO.
        """
        if self.config.device:
            return self.config.device

        if torch is not None and torch.cuda.is_available():
            return "cuda"
        if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        """
        Load YOLO model and move it to the chosen device.
        """
        self._device_str = self._select_device()
        logger.info("Loading YOLO model '%s' on device '%s'",
                    self.config.model_path, self._device_str)

        try:
            model = YOLO(self.config.model_path)
        except Exception as exc:
            logger.exception("Failed to load YOLO model from '%s'",
                             self.config.model_path)
            raise RuntimeError(f"Could not load YOLO model: {exc}") from exc

        if torch is not None and self._device_str == "cuda" and self.config.half:
            try:
                model.to("cuda")
                model.fuse()
                model.half()
                logger.info("YOLO model loaded in FP16 on CUDA")
            except Exception:
                logger.warning("FP16 initialization failed; falling back to FP32 on CUDA")
                model.to("cuda")
        else:
            try:
                model.to(self._device_str)
            except Exception:
                logger.warning("Failed to move model to '%s', falling back to CPU",
                               self._device_str)
                model.to("cpu")
                self._device_str = "cpu"

        self._model = model
        logger.info("YOLO model ready")

    @property
    def device(self) -> str:
        return self._device_str

    def warmup(self, img_shape: tuple[int, int, int] = (480, 640, 3)) -> None:
        """
        Optional warmup call to stabilize latency on the first frame.
        """
        if self._model is None:
            return

        dummy = np.zeros(img_shape, dtype=np.uint8)
        _ = self._run_yolo(dummy)

    def detect(self, frame: Frame | np.ndarray) -> List[Detection]:
        """
        Run person detection on a Frame or raw BGR image.

        Parameters
        ----------
        frame:
            - If Frame: uses frame.image (BGR numpy array).
            - If np.ndarray: must be HxWx3 BGR.

        Returns
        -------
        List[Detection]:
            All detections that match `config.classes` (by default, only people).
        """
        if isinstance(frame, Frame):
            img = frame.image
        else:
            img = frame

        if img is None:
            raise ValueError("Detector.detect() received empty image")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"Detector.detect() expects HxWx3 image, got shape {img.shape}"
            )

        results = self._run_yolo(img)
        detections: List[Detection] = []

        if results is None:
            return detections

        res = results[0]

        if not hasattr(res, "boxes") or res.boxes is None:
            return detections

        boxes = res.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        names = res.names

        for i in range(len(xyxy)):
            class_id = int(cls[i])
            score = float(conf[i])

            if self.config.classes is not None and class_id not in self.config.classes:
                continue

            x1, y1, x2, y2 = xyxy[i]
            class_name = names.get(class_id, str(class_id))

            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=score,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        return detections

    def _run_yolo(self, img: np.ndarray):
        """
        Internal helper to call Ultralytics YOLO with our config.
        """
        if self._model is None:
            raise RuntimeError("YOLO model is not loaded")

        try:
            results = self._model.predict(
                img,
                imgsz=self.config.imgsz,
                conf=self.config.conf,
                iou=self.config.iou,
                device=self._device_str,
                verbose=False,
                classes=None
            )
            return results
        except Exception as exc:
            logger.exception("YOLO inference failed: %s", exc)
            return None
