"""Production command-line entry point for the v2 measurement engine."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

from waitress import serve

from measure_api.v2.application import EngineApplication
from measure_api.v2.http import create_production_app
from measure_api.v2.image import DEFAULT_POOL_NAME, SharedMemoryImageReader


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the single-instance production HTTP adapter."""
    parser = argparse.ArgumentParser(description="Headless v2 measurement engine")
    parser.add_argument("--host", default=os.getenv("MEASURE_V2_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MEASURE_V2_PORT", "9004")),
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.getenv("MEASURE_V2_ASSET_ROOT", "/var/lib/measure-v2")),
    )
    parser.add_argument(
        "--shm-name",
        default=os.getenv("MEASURE_V2_SHM_NAME", DEFAULT_POOL_NAME),
    )
    args = parser.parse_args(argv)

    image_reader = SharedMemoryImageReader(args.shm_name)
    image_reader.ensure_pool()
    application = EngineApplication(args.asset_root)
    http = create_production_app(
        application,
        image_reader,
    )
    try:
        serve(http, host=args.host, port=args.port, threads=4)
    finally:
        application.close()


if __name__ == "__main__":
    main()
