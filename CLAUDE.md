# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Pure Python reimplementation of Halcon's 1D/2D measurement tools (edge detection, line/circle fitting) without the commercial Halcon library. The project has a **flat file structure** — the README describes a package layout (`halcon_1d_measure/`, `examples/`, `tests/`) that does not exist on disk. All code lives in top-level `.py` files.

## Build / test / run

No build step. Install dependencies:

```bash
pip install numpy opencv-python scipy matplotlib
```

Run tests:

```bash
# Run all tests with pytest
pytest test_1d_measure.py test_2d_measure.py -v

# Run a single test file
pytest test_1d_measure.py -v

# Run a specific test class/method
pytest test_1d_measure.py::TestHalcon1DMeasure::test_measure_pos_1 -v

# Run the 2D test as a script (has its own __main__)
python test_2d_measure.py
```

Run tests:

```bash
# All tests
pytest test_1d_measure.py test_2d_measure.py test_template_measure.py -v

# Template matching tests only
pytest test_template_measure.py -v

# With built-in test runner (visual demos included)
python test_template_measure.py
```

Run the 1D measure demo:

```bash
python measure1D.py
```

## Architecture

### `measure1D.py` — Core 1D edge detection

Contains `Halcon1DMeasure`, the single most important class in the project. Everything else depends on it.

**Algorithm pipeline** (`measure_pos` method):
1. Extract a slanted ROI via `cv2.warpAffine` into an axis-aligned rectangle of shape `(length2, length1)`
2. Average pixel values across each column (axis=0) → 1D gray profile
3. Gaussian smooth with `scipy.ndimage.gaussian_filter1d(sigma=...)`
4. Compute `np.gradient` → derivative signal
5. Find positive and negative peaks above threshold (simple local-max scan)
6. Subpixel refinement via quadratic polynomial fitting (`np.polyfit`) on a 3-sample window around each peak
7. Inverse affine transform back to original image coordinates
8. Optionally filter by `select` (`'first'`/`'last'`/`'all'`) and `transition` (`'positive'`/`'negative'`/`'all'`)

**Key coordinate conventions:**
- `angle`: radians, `0` = right, `π/2` = down (image row direction)
- `length1`: half-length along the measurement direction
- `length2`: half-width perpendicular to the measurement direction
- The README's diagram showing `length1` as the longer dimension and `length2` as the shorter one is correct

**Threshold modes:**
- `normalize_threshold=False` (default): threshold in `[0, 255]` — matches Halcon behavior
- `normalize_threshold=True`: threshold in `[0, 1]`, internally multiplied by 255 — the code calls this "more user-friendly"

**Debug mode** (`debug=True` in `measure_pos`): opens 6 sequential OpenCV windows showing the profile, smoothed profile, gradient, gradient with peaks, ROI with peaks, and original image with detected edges. Each window appears via `cv2.imshow` + `cv2.waitKey(wait_time)` — effective wait_time is 1000ms per window.

### `measure2D.py` — Geometric measurement (metrology)

Depends on `Halcon1DMeasure` from `measure1D.py`. Three classes:

- **`LineMeasureObject`**: Generates `num_measures` equally-spaced measurement rectangles perpendicular to a user-defined line segment. Each rectangle runs `measure_pos`, picks the edge closest to the line, then fits a line to all collected edge points via SVD. Returns line params `(a,b,c)` in `ax + by + c = 0` form, plus endpoints, angle, and fit error statistics.

- **`CircleMeasureObject`**: Generates `num_measures` measurement rectangles spaced around a circle, each pointing radially toward the center. Within each rectangle, `measure_pos` runs, edges are filtered by `[radius_min, radius_max]`, and the edge closest to the expected radius is selected. Fits a circle to all filtered edge points via algebraic least squares. The `radius_min`/`radius_max` filtering is critical — without it, inner/outer artifacts produce spurious edges.

- **`MetrologyModel`**: Container that manages multiple `LineMeasureObject` or `CircleMeasureObject` instances, batches `measure()` and `visualize()` calls. Objects are indexed by an auto-incrementing counter.

**Note on `length1`/`length2`** in measure2D: `LineMeasureObject` and `CircleMeasureObject` swap length1/length2 when constructing `Halcon1DMeasure` internally — their own `measure_length1` maps to Halcon1DMeasure's `length2` and vice versa (see [measure2D.py:131-136](measure2D.py#L131-L136)).

### `measure_template.py` — Template-matching point measurement

Self-contained module (no dependency on measure1D/measure2D). Implements a "teach once, measure many" workflow: define templates from a reference image, then locate them on new inspection images via NCC correlation.

**`Preprocessor` Protocol** — Pluggable image enhancement interface applied identically to template and inspection image:

```python
class Preprocessor(Protocol):
    name: str                                   # display label
    def serialize(self) -> dict: ...            # for .npz persistence
    @staticmethod
    def deserialize(data: dict) -> Preprocessor: ...
    def __call__(self, image: np.ndarray) -> np.ndarray: ...
```

Built-in implementations (all registered in `_PREPROCESSOR_REGISTRY`):

| Class | Key | Output dtype | Purpose |
|---|---|---|---|
| `RawPreprocessor` | `'raw'` | float32 | No enhancement (default) |
| `CannyPreprocessor(t1, t2)` | `'canny'` | uint8 (0/255) | Canny edge detection |
| `SobelPreprocessor(ksize)` | `'sobel'` | float32 | Sobel gradient magnitude |
| `CLAHEPreprocessor(clip_limit)` | `'clahe'` | float32 | Contrast-limited adaptive histogram equalization |
| `ThresholdPreprocessor(t, mode)` | `'threshold'` | uint8 (0/255) | Global threshold binarization |

Users register custom preprocessors via `_PREPROCESSOR_REGISTRY['my_type'] = MyClass`.

**`TemplatePoint`** — Single template point, following `LineMeasureObject`/`CircleMeasureObject` pattern:
- `__init__(reference_image, click_row, click_col, template_size=80, preprocessor=None, match_score_threshold=0.5, use_subpixel=True)` — crops template + applies preprocessor immediately. `preprocessor=None` defaults to `RawPreprocessor`.
- `measure(inspection_image, search_region=None) -> Dict` — applies same preprocessor to inspection image, runs `cv2.matchTemplate(TM_CCOEFF_NORMED)`, subpixel refinement via 2D quadratic fitting. Returns `{'matched_row', 'matched_col', 'dx', 'dy', 'match_score', 'valid', ...}`.
- `visualize(image, ...) -> np.ndarray` — draws template box + matched crosshair. `wait_time=-1` suppresses display.
- `save(filepath)` / `from_file(filepath, preprocessor=None)` — `.npz` serialization with JSON-encoded preprocessor data. `from_file` accepts optional `preprocessor` override. Backward-compatible with old `use_edges` format.

**`DistanceMeasure`** — Two-point container, following `MetrologyModel` pattern:
- `__init__(point_a, point_b)` / `measure(image) -> Dict` — returns `{'point_a', 'point_b', 'distance', 'valid'}`
- `visualize(image, ...)` — draws both points + distance line with label

### Test files

- **`test_1d_measure.py`**: pytest-based, `TestHalcon1DMeasure` class with fixtures for synthetic images and a real image loaded from `data/sample/bottleneck_2.jpg`. Tests cover measure_pos, measure_pairs, visualization, and threshold modes.

- **`test_2d_measure.py`**: Also pytest-compatible but has its own `run_all_tests()` runner and `__main__` block. Tests for line fitting, circle fitting, and MetrologyModel. The real-image tests reference image paths that may not exist locally.

- **`test_template_measure.py`**: pytest + built-in `run_all_tests()` runner. 28 tests across `TestTemplatePoint` (construction, matching, translation, subpixel, serialization), `TestDistanceMeasure` (distance accuracy, partial failure), `TestPreprocessor` (all 5 built-in preprocessors, serialization roundtrip, backward compat, custom registration), and two interactive visual demos (`test_visual_demo` with synthetic images, `test_visual_real_demo` with real images from `data/sample/`).

### Data directory

`data/` is gitignored. Test fixtures reference `data/sample/bottleneck_2.jpg` — a real product image used for integration tests. This file must be obtained separately.

### personal preferences
Prioritize answering in Chinese