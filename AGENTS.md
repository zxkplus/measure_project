# Repository Guidelines

## Project Structure & Module Organization

```
.
├── measure1D.py              # 1-D edge detection (Halcon1DMeasure)
├── measure2D.py              # 2-D fit: lines, circles, rectangles
├── measure_calibration.py    # Camera & stereo‑rig calibration
├── measure_template.py       # Template‑based matching
├── measure_workflow.py       # Orchestrated measurement workflow
├── multi_target_workflow.py  # Multi‑target detection pipeline
├── project_manager.py        # Project serialization / load
├── run_gui.py                # GUI entry point
├── measurement/              # Core library package
│   ├── constants.py, signal_ops.py, transforms.py, viz.py
├── measure_gui/              # PyQt / OpenCV GUI panels
│   ├── tool_panel.py, alignment.py, result_panel.py, ...
├── tests/                    # Pytest test suite
│   ├── conftest.py           # Shared fixtures (headless mode, output dirs)
│   └── test_*.py
├── data/sample/              # Sample images for testing
└── output/                   # Generated test outputs (gitignored)
```

The root modules (`measure*.py`) are standalone and importable directly. The `measurement/` package holds reusable primitives (signal processing, coordinate transforms, visualization). GUI code lives entirely inside `measure_gui/`.

## Build, Test, and Development Commands

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
