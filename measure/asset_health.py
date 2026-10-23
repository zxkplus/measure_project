"""Inspect a project/model directory and report what is missing or unusable.

Three entry points reach the same on-disk layout — the GUI, the v1 API, and the
engine directly — and they used to disagree about what is required. The v1 API
silently skipped a missing workflow file and left the caller with a ``None``
workflow that failed later for unrelated-looking reasons; the GUI aborted on a
missing ``project.json`` even though that file holds nothing but view state.

This module is the single answer to "what is in this directory and is it
loadable", so those three cannot drift apart again.

Two layouts exist, and they share their model asset:

* **model asset** — ``manifest.json`` + ``reference.npy`` (+ optional
  ``workflow.npz``).  Written by the engine.  This is the only source of truth
  for the ROI geometry, alignment mode, measurement objects and detector
  reference, and it is identical in both layouts.
* **GUI state** — ``project.json`` (view/zoom/layout/tool order) plus
  ``reference.png`` and ``inspection.png`` copies for portability.
* **API state** — ``config.json`` (measurement defs, phase) plus a
  ``template.npz`` cache written under a different name.

The historical difference is therefore *only* in the outer state files and the
cache filename, not in the model itself, which is why cross-loading is a matter of
falling back to the other state file rather than converting anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- filenames, mirroring the constants in the modules that own them ---------
# Imported lazily where a circular import would otherwise appear; kept as plain
# strings here because this module is deliberately dependency-free (both the GUI
# and the API import it, and neither should drag the other's imports in).

MODEL_MANIFEST = "manifest.json"
MODEL_REFERENCE = "reference.npy"
MODEL_DERIVED = "workflow.npz"

GUI_STATE = "project.json"
GUI_REFERENCE_PNG = "reference.png"
GUI_INSPECTION_PNG = "inspection.png"

API_STATE = "config.json"
API_DERIVED = "template.npz"

#: Contract versions this reader understands, as model-asset manifests carry them.
SUPPORTED_CONTRACT_VERSIONS = ("3.0", "3.1")

#: Format identifiers reported by :func:`inspect_asset`.
FORMAT_GUI = "gui"
FORMAT_API = "api"
FORMAT_MODEL_ONLY = "model_only"
FORMAT_LEGACY = "legacy_npz"
#: A modern project missing its manifest: has a cache but the cache is the current
#: format, so it cannot be migrated — the manifest must be restored.
FORMAT_DAMAGED = "damaged"
FORMAT_UNKNOWN = "unknown"


@dataclass
class AssetHealth:
    """What a directory contains and whether it can be opened.

    Attributes:
        directory: The inspected directory (``None`` when the path did not exist).
        exists: Whether the directory exists.
        format: One of the ``FORMAT_*`` constants.
        contract_version: Model-asset contract version, when a manifest is present.
        requires_reference_rebuild: ``reference.npy`` is absent but recoverable
            from ``reference.png``.
        missing_required: Files whose absence makes loading impossible.
        missing_optional: Files whose absence costs functionality or convenience
            but still allows loading.
        problems: Human-readable explanations, suitable for a dialog or a log.
    """

    directory: Optional[Path]
    exists: bool = False
    format: str = FORMAT_UNKNOWN
    contract_version: Optional[str] = None
    requires_reference_rebuild: bool = False
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def loadable(self) -> bool:
        """Whether the model can be opened at all.

        Optional files never block: every one of them has a defined fallback
        (a derived cache is recomputed, GUI state falls back to defaults, the API
        state falls back to the manifest).  Refusing to open a perfectly good
        model because a view-layout file is gone would be a worse failure than
        the one it reports.
        """
        return self.exists and not self.missing_required

    @property
    def is_model_asset(self) -> bool:
        """Whether a manifest model asset is present (regardless of state files).

        False for an untrained project — one whose state file exists but which has
        never been taught.  Callers must check this before asking the engine to load
        anything, because there is no manifest to load yet.
        """
        return self.contract_version is not None

    def summary(self) -> str:
        """One-line description for logs and status bars."""
        if not self.exists:
            return f"目录不存在: {self.directory}"
        if not self.loadable:
            return ("无法打开工程：缺少 " + "、".join(self.missing_required)
                    + f"（{self.directory}）")
        parts = [f"{self.format} 工程"]
        if self.contract_version:
            parts.append(f"契约 {self.contract_version}")
        if self.requires_reference_rebuild:
            parts.append("reference.npy 将自 PNG 重建")
        if self.missing_optional:
            parts.append("缺少 " + "、".join(self.missing_optional) + "（不影响打开）")
        return "；".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form, for API responses."""
        return {
            "directory": None if self.directory is None else str(self.directory),
            "exists": self.exists,
            "format": self.format,
            "contract_version": self.contract_version,
            "loadable": self.loadable,
            "requires_reference_rebuild": self.requires_reference_rebuild,
            "missing_required": list(self.missing_required),
            "missing_optional": list(self.missing_optional),
            "problems": list(self.problems),
            "summary": self.summary(),
        }


def _is_legacy_archive(path: Path) -> bool:
    """True when ``path`` is a pre-manifest single-file archive.

    Such an archive stores a ``workflow_meta`` array; the current format's cache
    stores only the derived arrays.  The distinction matters because only the former
    can be upgraded.
    """
    try:
        import numpy as np

        with np.load(str(path), allow_pickle=True) as data:
            return "workflow_meta" in data
    except Exception:  # noqa: BLE001 - an unreadable cache is not a legacy archive
        return False


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON object, or ``None`` when absent/unreadable/not an object."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def detect_format(directory: Path) -> Tuple[str, Optional[str]]:
    """Identify the layout and, for model assets, its contract version.

    Order matters: the model manifest decides whether this is an asset at all, and
    the presence of a state file decides which tool wrote it.  A directory with
    both state files is reported as GUI because that is the format the GUI reads
    first; the API falls back to ``config.json`` regardless of the label.

    Args:
        directory: Directory to classify.

    Returns:
        ``(format, contract_version)``.
    """
    manifest = _read_json(directory / MODEL_MANIFEST)
    if manifest is not None:
        version = manifest.get("contract_version")
        if not isinstance(version, str):
            version = None
        if (directory / GUI_STATE).is_file():
            return FORMAT_GUI, version
        if (directory / API_STATE).is_file():
            return FORMAT_API, version
        # A bare model asset: no tool has written its own state yet, but the
        # model is complete and loadable.
        return FORMAT_MODEL_ONLY, version

    # No model manifest.  Distinguish a genuine pre-manifest archive from a modern
    # project whose manifest was deleted: the former carries ``workflow_meta`` and
    # can be migrated, the latter is damage and migrating it would convert the loss
    # into a plausible-looking model.  Reading the archive is the only way to tell,
    # and it is cheap.
    for cache_name in (MODEL_DERIVED, API_DERIVED):
        cache = directory / cache_name
        if cache.is_file():
            return (FORMAT_LEGACY if _is_legacy_archive(cache)
                    else FORMAT_DAMAGED), None

    # A state file with no model is an untrained project, not a broken one: saving
    # before teaching anything is normal at the start of a session, and refusing to
    # reopen it would make the first save useless.
    if (directory / GUI_STATE).is_file() or (directory / API_STATE).is_file():
        return FORMAT_API if (directory / API_STATE).is_file() else FORMAT_GUI, None

    return FORMAT_UNKNOWN, None


def inspect_asset(path: str) -> AssetHealth:
    """Check whether ``path`` holds a loadable project and what is missing.

    Accepts a directory, a ``manifest.json``, or a cache ``.npz`` inside a project
    directory — the same three forms the engine's ``load()`` accepts — because
    callers receive whatever the user picked in a file dialog.

    Args:
        path: Directory or a file inside it.

    Returns:
        An :class:`AssetHealth` describing the directory.
    """
    target = Path(path)
    # A nonexistent path must not be silently reinterpreted as its parent: the
    # caller asked about something specific, and reporting on `/tmp` when they
    # typed `/tmp/typo` sends them looking in the wrong place.
    if not target.exists():
        health = AssetHealth(directory=target)
        health.problems.append(f"路径不存在: {target}")
        return health

    directory = target if target.is_dir() else target.parent
    health = AssetHealth(directory=directory)
    if not directory.is_dir():
        health.problems.append(f"目录不存在: {directory}")
        return health

    health.exists = True
    health.format, health.contract_version = detect_format(directory)

    if health.format == FORMAT_LEGACY:
        health.missing_required.append(MODEL_MANIFEST)
        health.problems.append(
            f"'{directory}' 是旧格式工程（只有 {MODEL_DERIVED}/{API_DERIVED}，"
            f"没有 {MODEL_MANIFEST}）。请运行 "
            f"scripts/migrate_models_to_manifest.py 迁移。"
        )
        return health

    if health.format == FORMAT_DAMAGED:
        health.missing_required.append(MODEL_MANIFEST)
        health.problems.append(
            f"无法打开：缺少必需文件 {MODEL_MANIFEST}。缓存文件是当前格式，"
            f"说明这是一个较新的工程而不是旧格式工程，无法通过迁移脚本恢复——"
            f"请从备份找回 {MODEL_MANIFEST}。"
        )
        return health

    if health.format == FORMAT_UNKNOWN:
        health.missing_required.append(MODEL_MANIFEST)
        health.problems.append(
            f"'{directory}' 不是工程目录：没有 {MODEL_MANIFEST}，"
            f"也没有可识别的旧格式文件。"
        )
        return health

    if health.contract_version is None:
        # No manifest: an untrained project.  Loadable — there is simply nothing
        # taught yet, which the caller discovers by the model being absent.
        health.missing_optional.append(MODEL_MANIFEST)
        return health

    if health.contract_version not in SUPPORTED_CONTRACT_VERSIONS:
        health.missing_required.append(MODEL_MANIFEST)
        health.problems.append(
            f"模型契约版本不受支持：{health.contract_version!r}。"
            f"当前支持 {', '.join(SUPPORTED_CONTRACT_VERSIONS)}。"
        )
        return health

    # --- model asset: the part that actually determines loadability ----------
    reference = directory / MODEL_REFERENCE
    reference_png = directory / GUI_REFERENCE_PNG
    if reference.is_file():
        pass
    elif reference_png.is_file():
        # Recoverable, not missing: the PNG is a copy of the same pixels written
        # for portability.  Reported separately so callers can rebuild it instead
        # of failing on a file that is reconstructible by construction.
        health.requires_reference_rebuild = True
        health.problems.append(
            f"{MODEL_REFERENCE} 缺失，将从 {GUI_REFERENCE_PNG} 重建。"
        )
    else:
        health.missing_required.append(MODEL_REFERENCE)
        health.problems.append(
            f"模型资产不完整：缺少 {MODEL_REFERENCE}，"
            f"且没有可用于重建的 {GUI_REFERENCE_PNG}。"
        )

    # --- optional: derived caches and per-tool state -------------------------
    has_cache = ((directory / MODEL_DERIVED).is_file()
                 or (directory / API_DERIVED).is_file())
    if not has_cache:
        # Search-free models legitimately have no cache; the manifest declares it
        # derived, so its absence is never a problem worth reporting as data loss.
        health.missing_optional.append(MODEL_DERIVED)

    if health.format == FORMAT_GUI:
        if not (directory / GUI_REFERENCE_PNG).is_file():
            health.missing_optional.append(GUI_REFERENCE_PNG)
    elif health.format == FORMAT_API:
        if not (directory / API_STATE).is_file():
            health.missing_optional.append(API_STATE)
    elif health.format == FORMAT_MODEL_ONLY:
        health.missing_optional.extend([GUI_STATE, API_STATE])

    return health


def message_for(health: "AssetHealth") -> str:
    """Human-readable explanation of an unhealthy asset.

    Prefers the specific diagnosis in ``problems`` over the generic "files are
    missing" sentence: a legacy asset needs a migration command, not a lecture
    about restoring files, and the two would otherwise be indistinguishable to a
    caller that only looked at ``missing_required``.
    """
    if health.problems:
        return " ".join(health.problems)
    return describe_missing(health.missing_required, health.missing_optional)


def describe_missing(required: List[str], optional: List[str]) -> str:
    """Build a message that names what is missing and what it costs.

    Separate from :meth:`AssetHealth.summary` because callers differ: the GUI shows
    a dialog and wants the consequence spelled out, while the API returns JSON and
    wants the lists.  Both word "required" the same way so users see one vocabulary.
    """
    if required:
        return ("无法打开：缺少必需文件 " + "、".join(required)
                + "。这些文件无法自动重建，请恢复它们或重新保存工程。")
    if optional:
        return ("可以打开，但缺少 " + "、".join(optional)
                + "（派生缓存会自动重建；界面状态将使用默认值）。")
    return "工程完整。"