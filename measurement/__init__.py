"""
Measurement package — portable, pure‑Python reimplementation of geometric
measurement tools (1‑D edge detection, line / circle fitting, template
matching).
"""

# ---------------------------------------------------------------------------
# Display‑mode control
# ---------------------------------------------------------------------------
# Call ``_apply_headless_patch()`` to make all ``cv2.imshow`` / ``cv2.waitKey``
# calls into no‑ops globally.  This is used by the test suite when
# ``--headless`` is passed.


def _apply_headless_patch() -> None:
    """Monkey-patch cv2 display functions into no‑ops (applied globally)."""
    import cv2 as _cv2

    _cv2.imshow = lambda *a, **kw: None  # type: ignore[assignment]
    _cv2.waitKey = lambda *a, **kw: -1   # type: ignore[assignment]
    _cv2.destroyAllWindows = lambda *a, **kw: None  # type: ignore[assignment]
    _cv2.destroyWindow = lambda *a, **kw: None  # type: ignore[assignment]
    _cv2.namedWindow = lambda *a, **kw: None  # type: ignore[assignment]
