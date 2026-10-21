"""Submit upstream-owned snapshots to V2 using only the Python standard library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional
import urllib.error
import urllib.request


class V2Client:
    """Small HTTP transport example; assets are exchanged through shared files."""

    def __init__(self, base_url: str, caller_id: str = "upstream-example") -> None:
        self.base_url = base_url.rstrip("/")
        self.caller_id = caller_id

    def request(self, route: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Read a public route or submit its JSON request and retain structured failures.

        Args:
            route: Public GET or POST route relative to the configured service base URL.
            payload: Optional JSON body; omitting it issues a read request.

        Returns:
            Parsed public JSON response; structured HTTP failures raise RuntimeError.
        """
        path = route if route.startswith("/api/v2/") else "/api/v2/" + route.lstrip("/")
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json", "X-Caller-Id": self.caller_id},
        )
        try:
            with urllib.request.urlopen(request, timeout=75) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            body = json.load(error)
            raise RuntimeError(json.dumps(body, ensure_ascii=False)) from error

    def calculate(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Poll one computation without resubmitting it or extending its work budget.

        Args:
            route: Public API route to read or submit through the configured HTTP base URL.
            payload: JSON-safe request or record to serialize without exposing partial content.

        Returns:
            Terminal public operation record after bounded polling of the original submission.
        """
        response = self.request(route, payload)
        deadline = time.monotonic() + payload.get("timeout_ms", 60000) / 1000 + 30
        while response["state"] in {"queued", "running", "cancelling"}:
            if time.monotonic() > deadline:
                raise TimeoutError("HTTP polling budget exceeded; query the retained operation")
            response = self.request(
                f"/operations/{response['operation_id']}?wait_ms=1000",
            )
        return response


def main() -> None:
    """Submit a saved request JSON and print the complete terminal operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9004")
    parser.add_argument("--caller-id", default="upstream-example")
    parser.add_argument("--route", required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    response = V2Client(args.base_url, args.caller_id).calculate(
        args.route, json.loads(args.request.read_text(encoding="utf-8")),
    )
    print(json.dumps(response, indent=2, ensure_ascii=False, allow_nan=False))
    if response["state"] != "succeeded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
