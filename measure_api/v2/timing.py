"""One Unix system monotonic epoch for cross-process deadline arbitration."""

import time


def monotonic_now() -> float:
    """Read the same non-wall-clock epoch in the HTTP owner and spawned workers.

    Python 3.9 on macOS gives each process a different time.monotonic() origin.
    Unix CLOCK_MONOTONIC provides a shared OS epoch, matching this runtime's flock
    and process-isolation deployment requirements.

    Returns:
        Seconds from the operating system's monotonic epoch.
    """
    return time.clock_gettime(time.CLOCK_MONOTONIC)
