"""Real HTTP acceptance server with an explicitly controlled OS publication boundary."""

import json
import os
from pathlib import Path

root = Path(os.environ["MEASURE_LIFECYCLE_ROOT"])
boundary = os.environ.get("MEASURE_LIFECYCLE_BOUNDARY")
original_link = os.link


def controlled_link(source, destination, *args, **kwargs):
    """Pause at an external filesystem boundary, never inside calculation logic."""
    name = Path(destination).name
    pause_before = ((boundary == "before-payload" and name == "preview.png")
                    or (boundary == "before-manifest" and name == "manifest.json"))
    pause_after = boundary == "after-manifest" and name == "manifest.json"
    if pause_before:
        (root / "boundary.json").write_text(json.dumps({"pid": os.getpid()}))
        with (root / "release.fifo").open("rb") as release:
            release.read(1)
    result = original_link(source, destination, *args, **kwargs)
    if pause_after:
        (root / "boundary.json").write_text(json.dumps({"pid": os.getpid()}))
        with (root / "release.fifo").open("rb") as release:
            release.read(1)
    return result


os.link = controlled_link

if __name__ == "__main__":
    import uvicorn

    from measure_api.v2.app import create_app
    from measure_api.v2.models import ServiceSettings

    settings = ServiceSettings(shared_roots={"exchange": root}, max_workers=1,
                               state_directory=root / "runtime", shutdown_wait_seconds=0)
    uvicorn.run(create_app(settings), host="127.0.0.1",
                port=int(os.environ["MEASURE_LIFECYCLE_PORT"]), log_level="error")
