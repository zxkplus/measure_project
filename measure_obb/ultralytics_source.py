"""Ultralytics YOLO-OBB detector adapter.

Implements the :class:`measure.obb_source.ObbSource` protocol on top of
ultralytics, so a learned detector can drive stage-1 localisation without the
engine ever importing torch.

Deliberately outside ``measure/``: importing this module pulls in ultralytics and
torch, which is a heavy, platform-specific dependency.  ``measure/`` stays
importable on a machine that has neither, and the GUI/API layers decide whether to
offer detector-based localisation at all.

Coordinate handling
-------------------
ultralytics reports boxes in ``(x, y)`` order while the engine works in
``(row, col)``.  Everything here converts at the boundary and nowhere else.

Angles are taken from ``xyxyxyxy`` (four corners) rather than ``xywhr``'s ``r``
field, for the reason given in :mod:`measure.obb_source`: the ``r`` convention has
moved between releases and depends on which edge the box was built from, whereas
four points are unambiguous.  ``r`` is available here only for diagnostics.

Usage::

    source = UltralyticsObbSource("bottle_obb.pt", conf=0.25, imgsz=1024)
    strategy = ObbAlignment()
    strategy.teach(reference_image, source=source, anchor_class_id=0)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from measure.obb_source import ObbObservation


def _sha256_file(path: str) -> str:
    """Digest a model file so the manifest can detect a swapped or edited weight."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UltralyticsObbSource:
    """An :class:`~measure.obb_source.ObbSource` backed by a YOLO-OBB model.

    Attributes:
        requires_roi: Always False — the detector finds its own boxes, which is
            what lets the GUI skip ROI drawing for this mode.
    """

    name = "ultralytics-obb"
    requires_roi = False

    def __init__(
        self,
        model_path: str,
        *,
        conf: float = 0.25,
        imgsz: int = 640,
        device: Optional[str] = None,
        class_names: Optional[Sequence[str]] = None,
        task: str = "obb",
    ):
        """Load a model.

        Args:
            model_path: Path to ``.pt`` or ``.torchscript`` weights.
            conf: Confidence threshold handed to the model.
            imgsz: Inference resolution.  Recorded in the manifest because
                changing it changes the detections, so a model reloaded at a
                different size is not the same model.
            device: Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.  ``None``
                lets ultralytics choose.
            class_names: Class labels, when known.  Detections carry names only as
                a convenience; selection uses the integer index.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            ImportError: If ultralytics is not installed.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "ultralytics is required for detector-based localisation. "
                "Install it in the environment that runs the detector; the "
                "measure/ engine itself does not need it."
            ) from exc

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"OBB model not found: {model_path}")

        self.model_path = str(model_path)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device
        self.task = task
        self._model = YOLO(self.model_path, task=task)
        # Prefer the checkpoint's own names; an explicit list wins so a project can
        # pin stable labels even if the weights are retrained.
        names = getattr(self._model, "names", None) or {}
        self._class_names: Dict[int, str] = (
            dict(enumerate(class_names)) if class_names
            else {int(k): str(v) for k, v in names.items()}
        )
        #: Digest of the weight file, so a manifest can prove which model ran.
        self.model_sha256 = _sha256_file(self.model_path)

    # -- inference ------------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[ObbObservation]:
        """Run the detector and convert its output to engine coordinates.

        Args:
            image: Grayscale or BGR image.  Grayscale is replicated to three
                channels because the detector expects three.

        Returns:
            One observation per detected box, in the model's own order.

        Raises:
            ValueError: If ``image`` is not a 2-D or 3-channel array.
        """
        prepared = self._prepare(image)
        kwargs: Dict[str, Any] = {"conf": self.conf, "imgsz": self.imgsz,
                                  "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self._model(prepared, **kwargs)
        if not results:
            return []
        return self._to_observations(results[0])

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        """Normalise an input image to the 3-channel form the model expects."""
        array = np.asarray(image)
        if array.ndim == 2:
            return np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim == 3 and array.shape[2] == 3:
            return array
        if array.ndim == 3 and array.shape[2] == 4:
            return array[:, :, :3]
        raise ValueError(
            f"expected a 2-D grayscale or 3-channel image, got shape {array.shape}"
        )

    def _to_observations(self, result: Any) -> List[ObbObservation]:
        """Convert one ultralytics ``Results`` object into observations."""
        obb = getattr(result, "obb", None)
        if obb is None or obb.xyxyxyxy is None or len(obb.xyxyxyxy) == 0:
            return []

        corners_xy = obb.xyxyxyxy.detach().cpu().numpy()
        classes = (obb.cls.detach().cpu().numpy().astype(int)
                   if obb.cls is not None else np.zeros(len(corners_xy), dtype=int))
        confidences = (obb.conf.detach().cpu().numpy()
                       if obb.conf is not None else None)

        observations: List[ObbObservation] = []
        for index, quad in enumerate(corners_xy):
            class_id = int(classes[index])
            score = None if confidences is None else float(confidences[index])
            # (x, y) -> (row, col).  from_corners recovers centre, size and angle
            # from the points, so no assumption is made about xywhr's `r`.
            corners_row_col = np.asarray(quad, dtype=np.float64)[:, ::-1]
            observations.append(
                ObbObservation.from_corners(
                    corners_row_col,
                    class_id=class_id,
                    score=score,
                    class_name=self._class_names.get(class_id),
                )
            )
        return observations

    # -- manifest support -----------------------------------------------------

    @property
    def class_names(self) -> Dict[int, str]:
        """Class index to label mapping from the checkpoint (or the override)."""
        return dict(self._class_names)

    def serialize(self) -> Dict[str, Any]:
        """Describe the model for the manifest.

        Includes the digest and inference settings, not just the path: a model
        reloaded from the same filename but different weights or resolution can
        produce different boxes, and a stored model that quietly changed is worse
        than one that fails to load.
        """
        return {
            "name": self.name,
            "path": self.model_path,
            "sha256": self.model_sha256,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "device": self.device,
            "task": self.task,
            "class_names": {str(k): v for k, v in self._class_names.items()},
        }

    def verify_matches(self, model_ref: Dict[str, Any]) -> Optional[str]:
        """Check a stored ``model_ref`` against this loaded model.

        Args:
            model_ref: The block previously produced by :meth:`serialize`.

        Returns:
            ``None`` when they agree, otherwise a human-readable description of
            the mismatch.  Returning the reason rather than raising lets a caller
            surface a warning and continue deliberately.
        """
        stored_path = model_ref.get("path")
        if stored_path and os.path.abspath(stored_path) != os.path.abspath(self.model_path):
            return (f"model path changed: manifest recorded '{stored_path}', "
                    f"loaded '{self.model_path}'")

        stored_sha = model_ref.get("sha256")
        if stored_sha and stored_sha != self.model_sha256:
            return (f"model digest changed for '{self.model_path}' "
                    f"(manifest {stored_sha[:12]}…, loaded {self.model_sha256[:12]}…)")

        stored_imgsz = model_ref.get("imgsz")
        if stored_imgsz is not None and int(stored_imgsz) != self.imgsz:
            return (f"inference size changed: manifest recorded {stored_imgsz}, "
                    f"loaded {self.imgsz}")

        return None