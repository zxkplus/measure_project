"""Load validated service-owned deployment settings without loading GUI dependencies."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from measure_api.v2.models import ServiceSettings


def load_settings(config: Optional[Union[str, Path]] = None) -> ServiceSettings:
    """Load a JSON settings file, or use the single shared-root convenience setting.

    Args:
        config: Explicit JSON filename; defaults to MEASURE_V2_CONFIG when supplied.

    Returns:
        Validated shared roots and bounded execution settings.
    """
    filename = config or os.getenv("MEASURE_V2_CONFIG")
    if filename:
        return ServiceSettings.model_validate_json(Path(filename).read_text(encoding="utf-8"))
    root = os.getenv("MEASURE_V2_SHARED_ROOT", "/var/lib/measure-v2/shared")
    return ServiceSettings(shared_roots={"shared": root})
