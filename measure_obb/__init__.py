"""Detector-backed localisation adapters.

Kept out of ``measure/`` on purpose: these modules import ML runtimes (torch,
ultralytics), which the measurement engine must not require.  ``measure`` exposes
the abstraction (``ObbSource``, ``ObbAlignment``); this package supplies concrete
detectors.

Importing this package therefore has a cost the engine does not.  Import the
specific adapter you need rather than the package, so a missing dependency is
reported by name::

    from measure_obb.ultralytics_source import UltralyticsObbSource
"""

from __future__ import annotations

__all__: list[str] = []
