"""Shared-directory resolution and immutable input preparation."""

from __future__ import annotations

import hashlib
import io
import struct
import uuid
import fcntl
import json
import os
import shutil
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator, Any, List, Tuple

import cv2
import numpy as np
from pydantic import BaseModel

from measure_api.v2.errors import ServiceError
from measure_api.v2.timing import monotonic_now
from measure_api.v2.models import (
    FileDescriptor, FileReference, ImageMetadata, OutputSpec, ServiceSettings,
)
from measure_api.v2.versions import decoder_identity


class SharedFiles:
    """Validate configured boundaries and read verified upstream inputs."""

    def __init__(self, settings: ServiceSettings, commit_guard: Any = None) -> None:
        self.settings = settings
        self.commit_guard = commit_guard
        self.pending_manifests = []
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}

    def resolve(self, reference: FileReference, write: bool = False) -> Path:
        """Resolve symlinks before permitting any access to a shared pathname.

        Args:
            reference: Shared pathname and optional expected content digest to verify.
            write: Whether resolution must also authorize a writable output root.

        Returns:
            Authorized resolved pathname; out-of-root or unwritable references raise errors.
        """
        roots = self.settings.shared_roots
        if reference.root is not None:
            if reference.root not in roots:
                raise ServiceError("file.unknown_root", "Unknown shared root alias")
            candidate = (roots[reference.root] / reference.path).resolve()
        else:
            candidate = Path(reference.path).resolve()
        aliases = [alias for alias, root in roots.items() if candidate.is_relative_to(root)]
        if not aliases:
            raise ServiceError(
                "file.outside_roots", "Path resolves outside configured shared roots",
            )
        if write:
            allowed = self.settings.writable_roots
            if allowed is not None and not set(aliases).intersection(allowed):
                raise ServiceError("file.output_forbidden", "Shared root does not permit outputs")
        return candidate

    def read_bytes(self, reference: FileReference) -> Tuple[bytes, FileReference]:
        """Read and hash one stable input, detecting rewritten or unsupported files.

        Args:
            reference: Shared pathname and optional expected content digest to verify.

        Returns:
            Stable input bytes and their pinned SHA-256 file reference.
        """
        path = self.resolve(reference)
        try:
            with path.open("rb") as source:
                before = os.fstat(source.fileno())
                if before.st_size > self.settings.max_input_bytes:
                    raise ServiceError("file.too_large", "Input exceeds configured file size limit")
                data = source.read(self.settings.max_input_bytes + 1)
                after = os.fstat(source.fileno())
        except FileNotFoundError as error:
            raise ServiceError(
                "file.not_found", "Referenced input file does not exist", 404,
            ) from error
        except (PermissionError, IsADirectoryError, OSError) as error:
            raise ServiceError("file.unreadable", "Referenced input cannot be read", 422) from error
        if len(data) > self.settings.max_input_bytes:
            raise ServiceError("file.too_large", "Input exceeds configured file size limit")
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            raise ServiceError("file.changed", "Input changed while being read", 409)
        digest = hashlib.sha256(data).hexdigest()
        if reference.sha256 is not None and reference.sha256.lower() != digest:
            raise ServiceError(
                "file.digest_mismatch", "Input does not match its expected SHA-256", 409,
            )
        return data, reference.model_copy(update={"sha256": digest})

    def read_image(self, reference: FileReference) -> Tuple[np.ndarray, FileReference]:
        """Decode only supported 8-bit gray or three-channel PNG/JPEG images.

        Args:
            reference: Shared pathname and optional expected content digest to verify.

        Returns:
            Independent decoded pixel array and its verified immutable source reference.
        """
        data, verified = self.read_bytes(reference)
        is_png = data.startswith(b"\x89PNG\r\n\x1a\n")
        is_jpeg = data.startswith(b"\xff\xd8\xff")
        if not (is_png or is_jpeg):
            raise ServiceError("image.unsupported_format", "Only PNG and JPEG inputs are supported")
        width, height = encoded_dimensions(data)
        if width * height > self.settings.max_image_pixels:
            raise ServiceError("image.too_large", "Image exceeds configured pixel limit")
        if width * height * 4 + len(data) > self.settings.max_worker_memory_bytes:
            raise ServiceError("operation.memory_limit", "Image exceeds worker memory budget", 429)
        identity = decoder_identity() + ":" + verified.sha256
        key = hashlib.sha256(identity.encode()).hexdigest()
        image = self._cached_image(key, data)
        if image is None:
            raise ServiceError("image.invalid", "Input cannot be decoded as an image")
        if image.dtype != np.uint8 or (image.ndim != 2 and image.shape[2] != 3):
            raise ServiceError("image.unsupported_pixels", "Expected GRAY8 or three-channel BGR8")
        if image.shape[0] * image.shape[1] > self.settings.max_image_pixels:
            raise ServiceError("image.too_large", "Image exceeds configured pixel limit")
        return image, verified

    def _cached_image(self, key: str, data: bytes) -> np.ndarray:
        """Cache disposable decoded pixels by verified content and decoder version."""
        enabled = (self.settings.cache_max_bytes > 0 and self.settings.cache_max_entries > 0
                   and self.settings.state_directory is not None)
        if not enabled:
            self.cache_stats["misses"] += 1
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        directory = self.settings.state_directory / "decoded-cache"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            for partial in directory.glob("*.partial"):
                partial.unlink(missing_ok=True)
            path = directory / (key + ".cache")
            if path.exists():
                try:
                    blob = path.read_bytes()
                    if hashlib.sha256(blob[32:]).digest() != blob[:32]:
                        raise ValueError("Cache checksum mismatch")
                    image = np.load(io.BytesIO(blob[32:]), allow_pickle=False)
                    os.utime(path, None)
                    self.cache_stats["hits"] += 1
                    return image.copy()
                except (ValueError, OSError):
                    path.unlink(missing_ok=True)
            self.cache_stats["misses"] += 1
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                return image
            buffer = io.BytesIO()
            np.save(buffer, image, allow_pickle=False)
            encoded = buffer.getvalue()
            blob = hashlib.sha256(encoded).digest() + encoded
            if len(blob) <= self.settings.cache_max_bytes:
                entries = sorted(directory.glob("*.cache"),
                                 key=lambda item: item.stat().st_mtime_ns)
                total = sum(item.stat().st_size for item in entries)
                while entries and (len(entries) >= self.settings.cache_max_entries
                                   or total + len(blob) > self.settings.cache_max_bytes):
                    oldest = entries.pop(0)
                    total -= oldest.stat().st_size
                    oldest.unlink()
                    self.cache_stats["evictions"] += 1
                stage = directory / (uuid.uuid4().hex + ".partial")
                try:
                    stage.write_bytes(blob)
                    os.replace(stage, path)
                finally:
                    stage.unlink(missing_ok=True)
            return image

    def finalize(self, outcome: dict) -> None:
        """Persist the validated result before atomically exposing its completion marker.

        Args:
            outcome: Validated worker result envelope and committed manifest references.
        """
        guard = self.commit_guard
        if guard is None:
            return
        directory = Path(guard.staging_directory).parent
        with filesystem_lock(guard.lock):
            check_commit(guard)
            completion = {"outcome": outcome, "manifests": [
                str(path) for _, path in self.pending_manifests]}
            atomic_json(directory / "completion.json", completion)
            for stage, path in self.pending_manifests:
                try:
                    os.link(stage, path)
                except FileExistsError as error:
                    raise ServiceError("output.conflict", "Completed output already exists", 409,
                    ) from error
                stage.unlink()
                sync_directory(path.parent)
            atomic_json(directory / "committed.json", {"committed": True})
            guard.committed.value = True

    def snapshot_request(self, value: Any) -> Any:
        """Pin every referenced input to content, leaving output directories untouched.

        Args:
            value: Supplied value to normalize under the public schema and coordinate
                convention.

        Returns:
            Equivalent request graph with every transitive input reference pinned.
        """
        if isinstance(value, OutputSpec):
            self.resolve(value.directory, write=True)
            return value
        if isinstance(value, FileReference):
            return self.read_bytes(value)[1]
        if isinstance(value, BaseModel):
            updates = {
                name: self.snapshot_request(getattr(value, name))
                for name in type(value).model_fields
            }
            return value.model_copy(update=updates)
        if isinstance(value, list):
            return [self.snapshot_request(item) for item in value]
        if isinstance(value, dict):
            return {key: self.snapshot_request(item) for key, item in value.items()}
        return value


def encoded_dimensions(data: bytes) -> Tuple[int, int]:
    """Check allocation size before OpenCV decodes compressed PNG/JPEG pixels.

    Args:
        data: Source payload to validate or transform without mutating the original.

    Returns:
        Declared width and height checked before compressed-image allocation.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"\xff\xd8"):
        offset = 2
        while offset + 4 <= len(data):
            if data[offset] != 255:
                break
            while offset < len(data) and data[offset] == 255:
                offset += 1
            marker = data[offset]
            offset += 1
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            if offset + 2 > len(data):
                break
            size = int.from_bytes(data[offset:offset + 2], "big")
            if size < 2 or offset + size > len(data):
                break
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF} and size >= 7:
                height, width = struct.unpack(">HH", data[offset + 3:offset + 7])
                return width, height
            offset += size
    raise ServiceError("image.invalid", "Input has no valid PNG/JPEG dimensions")


def image_metadata(image: np.ndarray, frame: str = "source") -> ImageMetadata:
    """Describe actual decoded data without trusting upstream dimensions.

    Args:
        image: Decoded pixel array whose actual dimensions and dtype are inspected.
        frame: Coordinate-frame identity for the returned image or mapping.

    Returns:
        Actual decoded width, height, channels, format and coordinate-frame identity.
    """
    return ImageMetadata(
        width=image.shape[1], height=image.shape[0],
        channels=1 if image.ndim == 2 else image.shape[2], dtype=str(image.dtype),
        pixel_format="GRAY8" if image.ndim == 2 else "BGR8", frame=frame,
    )


class ArtifactWriter:
    """Stage files on the destination filesystem and publish a final manifest last."""

    def __init__(
        self, files: SharedFiles, output: OutputSpec, operation_id: str,
    ) -> None:
        self.files = files
        self.output = output
        self.operation_id = operation_id

    def commit(
        self, payloads: List[Tuple[str, str, str, bytes]],
    ) -> Tuple[List[FileDescriptor], FileDescriptor]:
        """Commit (name, kind, format, bytes) payloads without overwriting any asset.

        Args:
            payloads: Artifact tuples of filename, semantic kind, format and immutable bytes.

        Returns:
            Committed artifact descriptors and their jointly published manifest descriptor.
        """
        guard = self.files.commit_guard
        if guard is not None:
            with filesystem_lock(guard.lock):
                check_commit(guard)
        return self._commit(payloads)

    def _commit(
        self, payloads: List[Tuple[str, str, str, bytes]],
    ) -> Tuple[List[FileDescriptor], FileDescriptor]:
        """Publish while the operation's cancellation gate is held."""
        directory = self.files.resolve(self.output.directory, write=True)
        namespace = directory / self.operation_id
        self.files.resolve(FileReference(path=str(namespace)), write=True)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            namespace.mkdir(exist_ok=False)
        except FileExistsError as error:
            raise ServiceError(
                "output.conflict", "Output attempt namespace already exists", 409,
            ) from error
        except OSError as error:
            raise ServiceError(
                "output.write_failed", "Output directory cannot be created", 422,
            ) from error
        guard = self.files.commit_guard
        if guard is not None:
            owned = Path(guard.staging_directory).parent / "owned-outputs"
            owned.mkdir(parents=True, exist_ok=True)
            atomic_json(owned / (hashlib.sha256(str(namespace).encode()).hexdigest() + ".json"),
                        {"path": str(namespace)})
        descriptors = []
        try:
            for name, kind, format_name, content in payloads:
                if Path(name).name != name or name in {"manifest.json", ".", ".."}:
                    raise ServiceError("output.invalid_name", "Invalid artifact name")
                descriptors.append(self._commit_file(namespace, name, kind, format_name, content))
            manifest = json.dumps({
                "schema_version": "1.0", "operation_id": self.operation_id,
                "files": [item.model_dump(mode="json") for item in descriptors],
            }, sort_keys=True, allow_nan=False).encode("utf-8")
            guard = self.files.commit_guard
            if guard is not None and guard.finalize:
                if self.files.pending_manifests:
                    raise ServiceError(
                        "output.multiple_commits", "Aggregate final artifacts in one commit", 409,
                    )
                marker = self._commit_file(
                    namespace, "manifest.json", "manifest", "json", manifest,
                                           deferred=True,
                )
            elif guard is not None:
                with filesystem_lock(guard.lock):
                    check_commit(guard)
                    marker = self._commit_file(
                        namespace, "manifest.json", "manifest", "json", manifest,
                    )
            else:
                marker = self._commit_file(
                        namespace, "manifest.json", "manifest", "json", manifest,
                    )
            return descriptors, marker
        except BaseException as error:
            shutil.rmtree(namespace, ignore_errors=True)
            if isinstance(error, OSError):
                raise ServiceError("output.write_failed", "Artifact commit failed", 422) from error
            raise

    def _commit_file(
        self, directory: Path, name: str, kind: str, format_name: str, data: bytes,
        deferred: bool = False,
    ) -> FileDescriptor:
        path = directory / name
        stage = directory / f".{name}.partial"
        with stage.open("xb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        if deferred:
            self.files.pending_manifests.append((stage, path))
        else:
            try:
                os.link(stage, path)
            except FileExistsError as error:
                raise ServiceError("output.conflict", "Completed output already exists", 409,
                    ) from error
            stage.unlink()
            sync_directory(directory)
        digest = hashlib.sha256(data).hexdigest()
        if self.output.directory.root is None:
            reference = FileReference(path=str(path), sha256=digest)
        else:
            root = self.files.settings.shared_roots[self.output.directory.root]
            reference = FileReference(
                root=self.output.directory.root, path=str(path.relative_to(root)), sha256=digest,
            )
        return FileDescriptor(kind=kind, format=format_name, file=reference,
                              sha256=digest, size_bytes=len(data))


@contextmanager
def filesystem_lock(path: str, blocking: bool = True) -> Iterator[None]:
    """Use a lock whose ownership the OS releases if a compute process dies.

    Args:
        path: Filesystem destination or diagnostic field path for this operation.
        blocking: Whether acquisition waits for an existing OS lock owner.

    Yields:
        None while the OS-owned exclusive lock protects the caller's block.
    """
    with Path(path).open("a+b") as lock:
        options = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(lock.fileno(), options)
        yield


def check_commit(guard: Any) -> None:
    """Reject a stopped or expired operation before any final publication.

    Args:
        guard: Shared deadline, stop flag and death-safe commit lock for publication.
    """

    if guard.stopped.value:
        raise ServiceError("operation.cancelled", "Operation stopped before commit", 409)
    if monotonic_now() >= guard.deadline:
        raise ServiceError("operation.timed_out", "Operation deadline exceeded before commit", 409)


def sync_directory(directory: Path) -> None:
    """Make file publication durable on its destination filesystem.

    Args:
        directory: Destination directory whose namespace publication must be flushed.
    """
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, payload: Any) -> None:
    """Commit an owned operational record without exposing partial JSON.

    Args:
        path: Filesystem destination or diagnostic field path for this operation.
        payload: JSON-safe request or record to serialize without exposing partial content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_suffix(".partial")
    with stage.open("w") as target:
        json.dump(payload, target, allow_nan=False)
        target.flush()
        os.fsync(target.fileno())
    os.replace(stage, path)
    sync_directory(path.parent)
