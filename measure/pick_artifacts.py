"""Debug artifacts for the gantry pick API.

A debug run is a self-contained directory holding the original images, an
annotated copy of every image, the request inputs, the full algorithm result,
the active recipe snapshot and a manifest.  The directory is written to a
temporary name and renamed only when complete, so readers never see a partial
run.

The module is intentionally free of HTTP and robot dependencies; the service
layer decides whether a run is recorded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from measure.circle_locator import CircleDetection

SCHEMA = "measure.pick_run"
VERSION = 1
SERVICE_VERSION = "pick-api/1.0.0"

RUN_ID_PATTERN = re.compile(
    r"^\d{8}-\d{6}-(?:vision|offset|locate)-[0-9a-f]{6}$"
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Return the local time with an explicit UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_run_id(run_type: str) -> str:
    """Build a sortable run identifier."""
    if run_type not in {"vision", "offset", "locate"}:
        raise ValueError(f"unknown run type: {run_type!r}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{run_type}-{uuid.uuid4().hex[:6]}"


def validate_run_id(run_id: str) -> bool:
    """Return whether *run_id* has the public format."""
    return bool(RUN_ID_PATTERN.fullmatch(str(run_id)))


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy values and paths to JSON-safe objects."""
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return to_jsonable(as_dict())
    return repr(value)


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically enough for a run directory under construction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def image_suffix(data: bytes, filename: Optional[str] = None) -> str:
    """Choose a stable suffix matching the bytes, falling back to the name."""
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:2] == b"BM":
        return ".bmp"
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            return ".jpg" if suffix == ".jpeg" else suffix
    return ".img"


def decode_image(data: bytes) -> Optional[np.ndarray]:
    """Decode uploaded bytes to a BGR image, or return ``None``."""
    if not data:
        return None
    buffer = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def resize_max(image: np.ndarray, max_px: int) -> Tuple[np.ndarray, float]:
    """Downscale so the long edge is at most *max_px*; return scale used."""
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_px:
        return image.copy(), 1.0
    scale = float(max_px) / float(long_edge)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _put_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float,
    colour: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw outlined text for readability on any background."""
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def render_annotation(
    image: np.ndarray,
    detection: Optional[CircleDetection] = None,
    *,
    index: Optional[int] = None,
    robot_xy: Optional[Sequence[float]] = None,
    principal_point: Optional[Sequence[float]] = None,
    extra_lines: Optional[Sequence[str]] = None,
    note: Optional[str] = None,
    max_px: int = 2000,
) -> np.ndarray:
    """Render a circle, centre cross and diagnostic labels on a copy.

    Args:
        image: BGR or grayscale source image.
        detection: Detection to draw; invalid detections are labelled.
        index: Optional zero-based image index.
        robot_xy: Flange XY recorded for this frame.
        principal_point: Camera centre to draw in blue.
        extra_lines: Additional text lines (already formatted).
        note: Optional error note shown in red.
        max_px: Long-edge limit of the source area.  The footer labels are
            added below the image, so total height can slightly exceed it.

    Returns:
        A BGR annotation image with a footer containing the labels.
    """
    source = image
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    canvas, scale = resize_max(source, max_px)
    height, width = canvas.shape[:2]
    base = max(1.0, min(2.5, max(height, width) / 1000.0))
    thickness = max(1, int(round(base * 2.0)))
    font_scale = max(0.45, min(1.4, base * 0.7))
    line_height = int(round(30 * font_scale + 8))

    lines: List[str] = []
    if index is not None:
        lines.append(f"#{index}")
    if detection is not None:
        centre = detection.center
        if detection.valid and all(math.isfinite(v) for v in centre):
            cx, cy = int(round(centre[0] * scale)), int(round(centre[1] * scale))
            radius = max(1, int(round(detection.radius * scale)))
            cv2.circle(canvas, (cx, cy), radius, (0, 255, 0), thickness, cv2.LINE_AA)
            arm = max(8, int(round(18 * base)))
            cv2.line(
                canvas, (cx - arm, cy), (cx + arm, cy),
                (0, 0, 255), thickness, cv2.LINE_AA,
            )
            cv2.line(
                canvas, (cx, cy - arm), (cx, cy + arm),
                (0, 0, 255), thickness, cv2.LINE_AA,
            )
            lines.append(
                f"uv=({centre[0]:.2f}, {centre[1]:.2f})  "
                f"r={detection.radius:.2f}px  conf={detection.confidence:.3f}"
            )
        else:
            lines.append(f"NO TARGET: {detection.reason or 'invalid'}")
    if robot_xy is not None and len(robot_xy) >= 2:
        lines.append(f"robot=({float(robot_xy[0]):.3f}, {float(robot_xy[1]):.3f})")
    if extra_lines:
        lines.extend(str(item) for item in extra_lines)
    if note:
        lines.append(str(note))

    if principal_point is not None and len(principal_point) >= 2:
        px = int(round(float(principal_point[0]) * scale))
        py = int(round(float(principal_point[1]) * scale))
        arm = max(8, int(round(16 * base)))
        cv2.line(
            canvas, (px - arm, py), (px + arm, py),
            (255, 0, 0), thickness, cv2.LINE_AA,
        )
        cv2.line(
            canvas, (px, py - arm), (px, py + arm),
            (255, 0, 0), thickness, cv2.LINE_AA,
        )
        if detection is not None and detection.valid:
            cv2.arrowedLine(
                canvas,
                (px, py),
                (int(round(detection.center[0] * scale)),
                 int(round(detection.center[1] * scale))),
                (0, 0, 255),
                max(1, thickness - 1),
                cv2.LINE_AA,
                tipLength=0.04,
            )

    # Put the labels in a footer strip.  A corner target would otherwise be
    # covered by the text, defeating the operator's visual check.
    if not lines:
        return canvas
    footer_height = line_height * len(lines) + 20
    panel = np.full(
        (height + footer_height, width, 3), (28, 28, 28), dtype=np.uint8
    )
    panel[:height] = canvas
    top = height + 8
    for line in lines:
        _put_text(
            panel,
            line,
            (10, top + line_height),
            font_scale,
            (255, 255, 255),
            max(1, thickness - 1),
        )
        top += line_height
    return panel


def make_overview(
    images: Sequence[np.ndarray],
    *,
    max_tile: int = 800,
    background: int = 240,
    gap: int = 14,
) -> np.ndarray:
    """Compose annotated images into a square-ish contact sheet."""
    tiles: List[np.ndarray] = []
    for image in images:
        tile, _ = resize_max(image, max_tile)
        tiles.append(tile)
    if not tiles:
        return np.full((16, 16, 3), background, dtype=np.uint8)

    columns = max(1, int(math.ceil(math.sqrt(len(tiles)))))
    rows = max(1, int(math.ceil(len(tiles) / columns)))
    cell_w = max(tile.shape[1] for tile in tiles) + gap
    cell_h = max(tile.shape[0] for tile in tiles) + gap
    canvas = np.full(
        (rows * cell_h + gap, columns * cell_w + gap, 3),
        background,
        dtype=np.uint8,
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        y = gap + row * cell_h + (cell_h - gap - tile.shape[0]) // 2
        x = gap + column * cell_w + (cell_w - gap - tile.shape[1]) // 2
        canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    return canvas


def save_jpeg(path: Path, image: np.ndarray, quality: int = 92) -> None:
    """Write a BGR image as JPEG, raising on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path), image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise RuntimeError(f"failed to write image: {path}")


# ---------------------------------------------------------------------------
# Run recorder
# ---------------------------------------------------------------------------


class RunRecorder:
    """Write one self-contained debug run.

    The directory is created under a hidden temporary name and renamed to
    ``<runs_dir>/<run_id>`` by :meth:`finish`, so a crash never leaves a
    half-written run that the listing endpoint would expose.
    """

    def __init__(
        self,
        runs_dir: Path,
        run_type: str,
        inputs: Dict[str, Any],
        *,
        recipe_snapshot: Optional[Dict[str, Any]] = None,
        annotated_max_px: int = 2000,
    ) -> None:
        if run_type not in {"vision", "offset", "locate"}:
            raise ValueError(f"unknown run type: {run_type!r}")
        self.run_id = new_run_id(run_type)
        self.run_type = run_type
        self.runs_dir = Path(runs_dir)
        self.recipe_snapshot = recipe_snapshot
        self.annotated_max_px = int(annotated_max_px)
        self.created_at = now_iso()
        self._started = time.perf_counter()
        self._finished = False

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir = self.runs_dir / self.run_id
        self.temp_dir = self.runs_dir / f".{self.run_id}.tmp"
        if self.final_dir.exists() or self.temp_dir.exists():
            raise RuntimeError(f"run already exists: {self.run_id}")
        (self.temp_dir / "images").mkdir(parents=True)

        self.inputs: Dict[str, Any] = dict(inputs)
        self.images: List[Dict[str, Any]] = []
        self._annotated: List[np.ndarray] = []
        self.overview_rel: Optional[str] = None

    def add_image(
        self,
        index: int,
        filename: Optional[str],
        data: bytes,
        image: Optional[np.ndarray],
        detection: Optional[CircleDetection] = None,
        *,
        name: Optional[str] = None,
        robot_xy: Optional[Sequence[float]] = None,
        principal_point: Optional[Sequence[float]] = None,
        extra_lines: Optional[Sequence[str]] = None,
        note: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save an original and an annotated copy, and return the manifest entry."""
        if self._finished:
            raise RuntimeError("run is already finished")
        stem = name or f"{int(index):02d}"
        suffix = image_suffix(data, filename)
        original_rel = f"images/{stem}_original{suffix}"
        annotated_rel = f"images/{stem}_annotated.jpg"
        original_path = self.temp_dir / original_rel
        annotated_path = self.temp_dir / annotated_rel
        original_path.parent.mkdir(parents=True, exist_ok=True)
        original_path.write_bytes(data)

        if image is None:
            placeholder = np.full((480, 640, 3), 235, dtype=np.uint8)
            _put_text(
                placeholder,
                "IMAGE UNREADABLE",
                (24, 60),
                1.0,
                (0, 0, 255),
                2,
            )
            annotated = placeholder
            width = height = 0
        else:
            annotated = render_annotation(
                image,
                detection,
                index=index,
                robot_xy=robot_xy,
                principal_point=principal_point,
                extra_lines=extra_lines,
                note=note or error,
                max_px=self.annotated_max_px,
            )
            height, width = image.shape[:2]

        save_jpeg(annotated_path, annotated)

        centre: Optional[List[float]] = None
        if detection is not None and all(
            math.isfinite(float(value)) for value in detection.center
        ):
            centre = [float(detection.center[0]), float(detection.center[1])]
        entry: Dict[str, Any] = {
            "index": int(index),
            "filename": filename or "",
            "original": original_rel,
            "annotated": annotated_rel,
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "width": int(width),
            "height": int(height),
            "uv": centre,
            "radius_px": None
            if detection is None
            else (float(detection.radius) if math.isfinite(detection.radius) else None),
            "confidence": None if detection is None else float(detection.confidence),
            "residual_px": None,
            "error": error,
        }
        self.images.append(entry)
        self._annotated.append(annotated)
        return entry

    def set_residual(self, index: int, residual_px: float) -> None:
        """Attach a fit residual to an image entry."""
        for entry in self.images:
            if int(entry["index"]) == int(index):
                entry["residual_px"] = float(residual_px)
                return

    def build_overview(self, *, max_tile: int = 800) -> Optional[str]:
        """Build ``overview.jpg`` from the annotated frames."""
        if not self._annotated:
            return None
        overview = make_overview(self._annotated, max_tile=max_tile)
        save_jpeg(self.temp_dir / "overview.jpg", overview)
        self.overview_rel = "overview.jpg"
        return self.overview_rel

    def finish(
        self,
        result: Dict[str, Any],
        *,
        status: str = "ok",
        error: Optional[Dict[str, Any]] = None,
        readme: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write all JSON files, the manifest, and publish the run directory.

        Args:
            result: Full algorithm result to store as ``result.json``.
            status: ``"ok"`` or ``"failed"``.
            error: Optional ``{"code": ..., "message": ...}`` dictionary.
            readme: Optional extra README text.

        Returns:
            The manifest dictionary.

        Raises:
            RuntimeError: If the run was already finished or publishing fails.
        """
        if self._finished:
            raise RuntimeError("run is already finished")
        if status not in {"ok", "failed"}:
            raise ValueError(f"unknown run status: {status!r}")

        write_json(self.temp_dir / "inputs.json", self.inputs)
        write_json(self.temp_dir / "result.json", result)
        if self.recipe_snapshot is not None:
            write_json(
                self.temp_dir / "recipe_snapshot.json",
                self.recipe_snapshot,
            )

        manifest = {
            "schema": SCHEMA,
            "version": VERSION,
            "run_id": self.run_id,
            "type": self.run_type,
            "status": status,
            "created_at": self.created_at,
            "duration_ms": int(round((time.perf_counter() - self._started) * 1000)),
            "error": to_jsonable(error),
            "inputs": "inputs.json",
            "result": "result.json",
            "recipe_snapshot": "recipe_snapshot.json"
            if self.recipe_snapshot is not None
            else None,
            "overview": self.overview_rel,
            "images": to_jsonable(self.images),
            "environment": environment_info(),
        }

        lines = [
            f"pick run {self.run_id}",
            f"type: {self.run_type}",
            f"status: {status}",
            f"created_at: {self.created_at}",
            "",
            "Files:",
            "  manifest.json          run metadata and file index",
            "  inputs.json            original request inputs",
            "  result.json            full algorithm output",
            "  recipe_snapshot.json   active recipe at run time",
            "  images/                original and annotated images",
            "  overview.jpg           contact sheet (vision calibration)",
            "",
            "Replay: python -m measure.pick_replay <path-to-this-run>",
        ]
        if readme:
            lines.extend(["", readme])
        (self.temp_dir / "README.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        write_json(self.temp_dir / "manifest.json", manifest)

        if self.final_dir.exists():
            raise RuntimeError(f"run already exists: {self.run_id}")
        self.temp_dir.rename(self.final_dir)
        self._finished = True
        return manifest

    def abort(self) -> None:
        """Remove the temporary directory if the run never completed."""
        if not self._finished:
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Environment and persistence helpers
# ---------------------------------------------------------------------------


def environment_info() -> Dict[str, Any]:
    """Return versions useful when reproducing a run."""
    return {
        "python": platform.python_version(),
        "opencv": str(cv2.__version__),
        "numpy": str(np.__version__),
        "platform": platform.platform(),
        "service": SERVICE_VERSION,
    }


def save_recipe_atomic(recipe: Any, path: Path) -> None:
    """Write a recipe to *path* via a temporary file and ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    recipe.save(str(temporary))
    os.replace(temporary, path)


def read_manifest(run_dir: Path) -> Dict[str, Any]:
    """Read and validate a run manifest."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    data = read_json(manifest_path)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"not a pick run manifest: {manifest_path}")
    return data


def list_runs(runs_dir: Path, limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent run summaries, newest first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    summaries: List[Dict[str, Any]] = []
    for entry in runs_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if not validate_run_id(entry.name):
            continue
        try:
            manifest = read_manifest(entry)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        images = manifest.get("images") or []
        annotated = next(
            (
                image.get("annotated")
                for image in images
                if image.get("annotated")
            ),
            None,
        )
        summaries.append(
            {
                "run_id": entry.name,
                "type": manifest.get("type"),
                "status": manifest.get("status"),
                "created_at": manifest.get("created_at"),
                "overview": manifest.get("overview"),
                "annotated": annotated,
            }
        )
    summaries.sort(
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("run_id") or ""),
        ),
        reverse=True,
    )
    return summaries[: max(0, int(limit))]


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO timestamp, returning ``None`` on failure."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def cleanup_runs(
    runs_dir: Path,
    *,
    keep_days_by_type: Optional[Dict[str, int]] = None,
    keep_max_by_type: Optional[Dict[str, int]] = None,
    protected: Iterable[str] = (),
) -> List[str]:
    """Delete old runs according to per-type age and count limits.

    Args:
        runs_dir: Root containing run directories.
        keep_days_by_type: Maximum age per ``vision``/``offset``/``locate``.
        keep_max_by_type: Maximum number of runs per type.
        protected: Run IDs that must never be deleted (active recipe).

    Returns:
        The run IDs that were removed.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    keep_days = dict(keep_days_by_type or {})
    keep_max = dict(keep_max_by_type or {})
    protected_set = {str(item) for item in protected if item}
    now = datetime.now().astimezone()
    grouped: Dict[str, List[Tuple[datetime, Path]]] = {}
    for entry in runs_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.name in protected_set or not validate_run_id(entry.name):
            continue
        try:
            manifest = read_manifest(entry)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        created = parse_timestamp(manifest.get("created_at"))
        if created is None:
            continue
        run_type = str(manifest.get("type") or "locate")
        grouped.setdefault(run_type, []).append((created, entry))

    removed: List[str] = []
    for run_type, items in grouped.items():
        items.sort(key=lambda pair: pair[0], reverse=True)
        max_keep = keep_max.get(run_type)
        max_age = keep_days.get(run_type)
        for index, (created, entry) in enumerate(items):
            too_many = max_keep is not None and index >= int(max_keep)
            too_old = max_age is not None and (now - created) > timedelta(days=int(max_age))
            if not (too_many or too_old):
                continue
            try:
                shutil.rmtree(entry)
            except OSError:
                continue
            removed.append(entry.name)
    return removed


def zip_run(run_dir: Path) -> Path:
    """Create a temporary ZIP archive containing a complete run directory."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run not found: {run_dir}")
    handle = tempfile.NamedTemporaryFile(
        prefix=f"{run_dir.name}-", suffix=".zip", delete=False
    )
    handle.close()
    archive = Path(handle.name)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(run_dir).as_posix())
    return archive


__all__ = [
    "RunRecorder",
    "cleanup_runs",
    "decode_image",
    "environment_info",
    "image_suffix",
    "list_runs",
    "make_overview",
    "new_run_id",
    "now_iso",
    "read_json",
    "read_manifest",
    "render_annotation",
    "resize_max",
    "save_jpeg",
    "save_recipe_atomic",
    "sha256_bytes",
    "to_jsonable",
    "validate_run_id",
    "write_json",
    "zip_run",
]
