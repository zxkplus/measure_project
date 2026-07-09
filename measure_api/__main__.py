"""
Entry point:  ``python -m measure_api``
"""

from __future__ import annotations

import argparse

from measure_api.config import Config
from measure_api.logger import setup_logging, get_logger
from measure_api.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure API Server")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port (overrides config)")
    parser.add_argument("--host", type=str, default=None,
                        help="Server host (overrides config)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom config file")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    args = parser.parse_args()

    cfg = Config.load(args.config)

    # CLI overrides
    if args.port is not None:
        cfg.set("server.port", args.port)
    if args.host is not None:
        cfg.set("server.host", args.host)
    if args.debug:
        cfg.set("server.debug", True)

    setup_logging(cfg.get("log"))
    logger = get_logger("main")

    app = create_app(cfg)

    host = cfg.get("server.host", "0.0.0.0")
    port = cfg.get("server.port", 5000)
    debug = cfg.get("server.debug", False)

    logger.info("Starting Measure API on %s:%d (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    main()
