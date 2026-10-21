"""One actual numerical implementation and runtime identity for archived calculations."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
import platform
from pathlib import Path
from typing import Dict, Literal

import cv2
import numpy as np
import scipy

from measure_api.v2.models import StrictModel

ALGORITHM_VERSION = "measure-v2-1"


class RuntimeIdentity(StrictModel):
    """Observed producer identity; unknown historical producers remain unspecified."""

    schema_version: Literal["1.0"] = "1.0"
    algorithm_version: str
    implementation_sha256: str
    python_version: str
    python_implementation: str
    architecture: str
    dependencies: Dict[str, str]


def runtime_dependencies() -> Dict[str, str]:
    """Observe imported numeric binaries and installed service distributions.

    Returns:
        Observed Python/numeric/service versions used for calculation provenance.
    """
    dependencies = {
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
    }
    for name in ("opencv-python-headless", "opencv-python", "fastapi", "pydantic", "uvicorn"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            continue
    return dependencies


def decoder_identity() -> str:
    """Namespace cached pixels by actual decoding runtime and fixed decode parameters.

    Returns:
        Cache namespace including actual decoder versions and fixed decode parameters.
    """
    dependencies = runtime_dependencies()
    relevant = {
        name: value
        for name, value in dependencies.items()
        if name == "numpy" or name.startswith("opencv")
    }
    return json.dumps(
        {
            "algorithm_version": ALGORITHM_VERSION,
            "decoder": "opencv.imdecode",
            "flags": int(cv2.IMREAD_UNCHANGED),
            "dependencies": relevant,
        },
        sort_keys=True,
    )


def runtime_identity() -> RuntimeIdentity:
    """Fingerprint shipped Python algorithms and the actually imported numeric runtime.

    Returns:
        Exact producer identity. Replay requires equality; no cross-version accuracy
        guarantee is inferred from a dependency's major or minor version.
    """
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for directory in (root / "measure", root / "measure_api" / "v2"):
        for source in sorted(directory.rglob("*.py")):
            digest.update(str(source.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(source.read_bytes())
            digest.update(b"\0")
    return RuntimeIdentity(
        algorithm_version=ALGORITHM_VERSION,
        implementation_sha256=digest.hexdigest(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        architecture=platform.machine(),
        dependencies=runtime_dependencies(),
    )
