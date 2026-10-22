# Repository Guidelines

## Project Structure & Module Organization

```
.
├── measure/                    # Unified measurement engine (imports no GUI/API)
│   ├── __init__.py             # Module initialization, exports main classes
│   ├── constants.py            # Constants (EPS, etc.)
│   ├── signal_ops.py           # Signal processing utilities
│   ├── transforms.py           # Coordinate transformations
│   ├── viz.py                  # Visualization helpers
│   ├── geometric_fits.py       # Reusable line/circle least-squares fits
│   ├── measure1D.py            # 1-D edge detection (Halcon1DMeasure)
│   ├── measure2D.py            # 2-D fit: lines, circles, rectangles
│   ├── measure_calibration.py  # Camera & stereo‑rig calibration
│   ├── pixel_calibration.py    # Scale / mm-per-pixel calibration
│   ├── measure_template.py     # Template‑based matching (TemplatePoint)
│   ├── measure_workflow.py     # Orchestrated measurement workflow
│   ├── multi_target_workflow.py # THE multi-target engine (only implementation)
│   ├── object_specs.py         # Single registry for all 10 object types
│   ├── alignment.py            # AlignmentStrategy: pluggable ROI locators
│   ├── obb_source.py           # OBB detection primitives (no ML dependency)
│   ├── rectify.py              # Pure geometry: crop_and_straighten, affine maps
│   └── result_format.py        # Result formatting shared by GUI and API layers
├── run_gui.py                  # GUI entry point
├── measure_gui/                # tkinter / OpenCV GUI panels
│   ├── app.py, template_view.py, image_canvas.py, tool_panel.py,
│   ├── project_manager.py, result_panel.py, calibration_dialog.py, dialogs.py
│   └── alignment.py, multi_target.py, utils.py    # thin shims over measure/*
├── measure_api/                # REST APIs — two generations, see "APIs" below
│   ├── server.py, project.py, quality.py, schemas.py, visualizer.py   # v1
│   ├── v2/                     # stateless FastAPI service (operation families)
│   └── measure-api/SKILL.md    # agent-facing docs + references/
├── scripts/                    # migrate_models_to_manifest.py, gen_api_reference.py, …
├── tests/                      # Pytest test suite (+ tests/v2 for the FastAPI service)
│   └── conftest.py             # Shared fixtures (headless mode, output dirs)
├── data/sample/                # Sample images for testing (gitignored)
└── output/                     # Generated test outputs (gitignored)
```

## APIs

Two generations coexist and are **not interchangeable**:

- **v1** — `measure_api/server.py`: Flask, stateful sessions under `/api/session/*`,
  persists `config.json` + `template.npz` + `reference.png`.
- **v2** — `measure_api/v2/`: FastAPI, **stateless** atomic operations under `/api/v2/*`,
  organised as operation families with typed port compatibility. Its authoritative contract
  is `docs/v2-openapi.json` (`docs/v2-api-reference.md` for field-level detail).

Neither API imports `measure_gui`; both use the shared `measure/` engine. Before working on
anything that crosses the two, read
[`measure_api/measure-api/references/v1-v2-divergence.md`](measure_api/measure-api/references/v1-v2-divergence.md)
— where they disagree, known shared defects, and undecided ownership questions.

Model assets taught by the GUI use manifest contract 3.1 (`manifest.json` + `reference.npy`
+ derived `workflow.npz`); legacy single-file `.npz` models upgrade automatically on open,
and 3.0 assets load unchanged. Detector-based localisation (`alignment.mode == "obb"`) writes
no template cache and records the detector's identity in `roi_state.model_ref`; the detector
itself is injected at run time, so `measure/` never imports torch.

The `measure/` package consolidates all measurement-related code into a single, portable module. It includes:
- **Core utilities**: constants, signal processing, coordinate transforms, visualization
- **1D measurement**: Halcon-style 1D edge detection
- **2D measurement**: Line, circle, and rectangle fitting
- **Calibration**: Camera and stereo rig calibration
- **Template matching**: Point-based template matching with subpixel refinement
- **Workflow**: Composable measurement workflow system
- **Multi-target**: Multi-target detection and measurement pipeline

Import examples:
```python
from measure import Halcon1DMeasure, LineMeasureObject, CircleMeasureObject
from measure import CameraCalibration, StereoRigCalibration
from measure import TemplatePoint, DistanceMeasure
from measure import MeasurementWorkflow, MultiTargetWorkflow
```
| Command | Purpose |
|---|---|
| `python run_gui.py` | Launch the measurement GUI application. |
| `pytest` | Run the full test suite. |
| `pytest --headless` | Run tests without `cv2.imshow` pop‑ups (default in `pyproject.toml`). |
| `pytest tests/test_1d_measure.py -v` | Run a specific test file with verbose output. |
| `pytest -k test_fit_circle` | Run tests matching a keyword expression. |

Tests write visual output to `tests/output/<test_name>/`. All test artifacts are gitignored.

## Coding Style & Naming Conventions

- **Language:** Python 3.9+.
- **Indentation:** 4 spaces; no tabs.
- **Naming:**
  - `PascalCase` for classes and test classes.
  - `snake_case` for functions, methods, and variables.
  - `UPPER_SNAKE_CASE` for module‑level constants.
- **Docstrings:** Google style (triple‑double‑quoted, with `Args:`, `Returns:` sections).
- **Imports:** Group order: standard library → third‑party → local. Separate groups with a blank line.
- **Type hints:** Required for public function signatures; encouraged internally.
- **Line length:** ≤ 100 characters (flake8 convention).

## Testing Guidelines

- **Framework:** pytest (configured in `pyproject.toml`).
- **Location:** All tests live under `tests/`, named `test_<module>.py`.
- **Naming:** Test classes are `Test<ClassName>`; test functions are `test_<behavior>`.
- **Fixtures:** Shared fixtures go in `tests/conftest.py`. The `test_output_dir` fixture creates a per‑test output directory automatically.
- **Coverage:** Aim for edge‑case coverage on fitting algorithms (e.g., degenerate point sets, empty inputs, extreme aspect ratios). Run with `pytest --cov=.` to check coverage.
- **Headless mode:** Tests must never block on GUI calls. Use the `display_or_save()` helper from `conftest.py` for any visual output.

## Commit & Pull Request Guidelines

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add visibility toggle (☑/☐) for each measurement tool
fix: correct circle measurement rectangle orientation
refactor(phase5): test infrastructure overhaul — remove pop‑ups, unify output
test: add calibration module tests (30 tests)
```

- Use scopes like `(phaseN)`, `(gui)`, or `(calibration)` when the change is scoped to a specific area.
- Keep the description imperative and concise (fifty‑character title preferred).
- If the PR closes an issue, reference it in the body: `Closes #123`.
- A PR description should summarize *what* changed and *why*. Include screenshots for any visual or GUI change.

## Agent‑Specific Instructions

This repository contains `.agents/`, `.mimocode/`, and `.claude/` directories that store per‑agent guidance and plans. When making changes:

- **Read first:** Check `CLAUDE.md` before editing any source file — it captures project conventions and current phase context.
- **Simulate before write:** For refactoring or core algorithm changes, prefer writing a focused test to validate behavior before modifying production code.
- **Output hygiene:** Never commit test artifacts (they are already gitignored). Place generated images in `output/` or `tests/output/`.
- **Language:** Keep commit messages and documentation in English. Inline comments and public docstrings should be English; internal implementation notes may use Chinese where it aids clarity.
