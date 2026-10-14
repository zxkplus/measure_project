"""Production command-line entry point for the FastAPI V2 computation service."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

import uvicorn

from measure_api.v2.app import create_app
from measure_api.v2.configuration import load_settings


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run one ASGI owner with independently bounded CPU processes.

    Args:
        argv: Optional command-line arguments; process arguments are used when omitted.
    """
    parser = argparse.ArgumentParser(description="Headless FastAPI V2 measurement service")
    parser.add_argument("--host", default=os.getenv("MEASURE_V2_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MEASURE_V2_PORT", "9004")))
    parser.add_argument(
        "--config", type=Path, default=os.getenv("MEASURE_V2_CONFIG"),
        help="JSON ServiceSettings file (shared_roots, writable_roots, state and limits)",
    )
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
