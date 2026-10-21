# v1 / v2 Divergence Report

**Audience: an AI agent or engineer working on either API generation.** Read this
before changing shared code, porting a model, or writing call code against the other
generation. It lists where the two implementations disagree *today*, which side owns
each difference, and what is still unresolved.

Generated from the merge of `measure_project/main` (tangzhihao's stateless v2 rewrite)
into the unified-engine branch. Every claim below was verified by executing code; the
method is stated so you can re-check it rather than trusting this file.

---

## 0. The one-line architecture difference

| | v1 (`/api/session/*`) | v2 (`/api/v2/*`) |
|---|---|---|
State | **stateful session**, phase state machine | **stateless** atomic operations |
Framework | Flask + waitress | FastAPI + pydantic + uvicorn |
Images | server-side absolute paths | upstream-owned `{root, path}` refs + sha256 |
Modeling | ROI + tools live in a session, saved to disk | no session; `/template-builds` publishes artifacts |
Orchestration | fixed pipeline inside `MultiTargetWorkflow` | caller composes ops; `recipes` gives a typed DAG |
Object types | 10 in `measure/object_specs.py` | 55 registered operations across `families/*.py` |
On-disk asset | `manifest.json` + `reference.npy` + `workflow.npz` (contract 3.0) | its own artifact sets; **does not read or write manifest assets** |

The two are **not interchangeable**. A v1 project directory cannot be loaded by v2 and
vice versa. Do not "fix" this by adding a converter unless someone asks for one — the
formats encode different assumptions about who owns the image files.

---

## 1. 🔴 Confirmed defect affecting both generations

### `Halcon1DMeasure.measure_pairs()` silently ignores its `transition` parameter

`measure/measure1D.py` calls the inner detector with both filter arguments hard-coded:

```python
self.measure_pos(img, sigma, threshold, "all", "all", False, debug)
#                                          ↑transition  ↑select
```

Signature is `measure_pos(img, sigma, threshold='30', transition='all', select='all', ...)`.
So `measure_pairs(..., transition=...)` accepts the argument and then discards it.

**Why it looks like it works:** pairing happens downstream, so results still differ per
`transition` value. Measured on an 8-band alternating image where `measure_pos` correctly
returns all=7 / positive=4 / negative=3 edges:

```
select=all  transition=positive -> centers=(14.5, 34.5, 54.5)
select=all  transition=negative -> centers=(24.5, 44.5, 64.5)
select=all  transition=all      -> centers=(14.5, 34.5, 54.5)   ← identical to positive
```

`all` must be a superset of `positive`. It is not, because no polarity filtering occurred.
**Any case where positive and negative transitions are unequal produces wrong pair counts.**

Who is affected:

- **v2**: `POST /api/v2/measurements/edge-pairs` exposes `polarity: Literal['positive',
  'negative', 'all']` to callers and passes it straight through
  (`measure_api/v2/families/measurements.py:263`). Any non-`all` request does not behave
  as documented.
- **v1 / GUI**: `EdgePairObject` forwards `self.transition` (`measure_workflow.py:1161`),
  so it is equally broken. Currently latent — all 8 shipped models use `transition=all`.

Suggested fix: pass both parameters through. Add a regression test asserting that
`transition="all"` returns at least as many pairs as `"positive"` and `"negative"` combined.

---

## 2. 🟠 Unresolved semantic conflict: circle into a point slot

The two generations reach **opposite conclusions** on the same input.

| | rule | `point ← circle` |
|---|---|---|
| v2 | `recipe_graph.accepts()`, comparing pydantic classes `PositionedPoint` vs `Circle` | **rejected** |
| v1 | `measure_api/schemas.TYPE_COMPATIBILITY` | **accepted** |

Verified directly: `accepts(PORT_TYPES['point'], PORT_TYPES['circle']) == False`.

Why v1 allows it:

1. `measure.measure_workflow.CircleResult` defines `row` and `col` properties commented
   *"Compatibility alias: row -> center_row"* — designed to satisfy point consumers.
2. The GUI dependency picker (`measure_gui/tool_panel.py:544`) filters candidates only for
   `EdgePair`; every other tool type is offered for every slot, so a user can legitimately
   wire a `FitCircle` into `point_a_label`.
3. Existing tests do exactly that (`measure_api/tests/test_project.py::test_add_composed`:
   `PointCircleDistance` with `point_label` → a `FitCircle`).

Correction to an earlier assumption: **no shipped model depends on this.** Scanning all 8
`models/*/manifest.json` found composed dependencies using only `pair_1_A` / `pair_1_B`.
The compatibility exists for the aliases, the GUI, and the tests — not for stored data.

Consequence: a recipe that v1 accepts, v2 rejects. Until this is decided, do not assume a
model taught in the GUI will validate in a v2 recipe.

Decision needed from the v2 owner: allow `Circle` on point ports (take the centre), or
tighten `TYPE_COMPATIBILITY` plus the GUI picker — the latter changes behaviour for
existing projects.

---

## 3. 🟡 Parameter defaults that do not migrate 1:1

Do not copy a template configuration between generations expecting the same behaviour.

| Concept | v1 / engine (`measure/measure_template.py`) | v2 (`families/templates.py`, `TemplateSearch`) |
|---|---|---|
Score gate | `match_score_threshold` default **0.5** | `score_threshold` default **0.7** |
Two-pass search | `coarse_fine` default **True** | `coarse_fine` default **False** |
Pyramid control | `pyramid_decimate` — 0 = disabled, 1 = downsample 2×, 2 = 4× … | `pyramid_levels` — 0..8 level count |
Angle range | `angle_range` stored as a **half-range** pair, e.g. `[-15, 15]` | `angle_min` / `angle_max` explicit bounds |
Scale search | `scale_invariant`, `scale_range`, `scale_step` | `scale_min` / `scale_max` / `scale_step`, plus `coarse_scale_step` |

`pyramid_decimate` and `pyramid_levels` are **different quantities**, not renamings: a
value of 2 means "quarter resolution" in one and "two levels" in the other. Never transfer
that number directly.

Note also that v2 validates ranges (`ge`/`le` constraints). Values outside them fail with
HTTP 422 rather than being clamped.

---

## 4. 🔵 Capability gaps (one direction each)

**Only v2 has these:**

- `frame` / `unit` on every observation, with `compatible()` refusing mixed-frame maths and
  `check_metric_frame()` requiring calibration to describe the measurement frame. Neither
  the engine result dicts nor v1 carry frame or unit at all (`grep frame\|unit` over
  `measure/result_format.py` and `measure/multi_target_workflow.py` returns nothing). This
  is a correctness gap for multi-camera or calibrated work, not a style preference.
- Scale-invariant search as a first-class operation option.
- Idempotency keys, operation cancellation, bounded spawn processes, artifact manifests with
  content hashing.
- Roundness reported explicitly as `sampled_diameter_spread`, documented in its own docstring
  as *"2*(max sampled radius - min sampled radius), not ellipse roundness"*.

**Only v1 / the engine has these:**

- Composed objects resolved internally by label, including `EdgePair` virtual endpoints
  `<label>_A` / `<label>_B`. In v2 these must be wired explicitly through recipe ports.
- Multi-target detection returning top-k instances from one call. v2's `/template-matches`
  returns matches but there is no equivalent single "detect then measure everything" call.
- Manifest v3.0 model assets consumable by the GUI, v1, and the migration script.

Both sides compute `ellipticity` / `sampled_diameter_spread` with the same formula
(`2 * (max_radius - min_radius)`); only v2 names it in a way that warns it is not true
roundness. If a customer acceptance criterion involves roundness, confirm which definition
is intended.

---

## 5. What changed in the shared engine (affects both)

These came from the unification work and are already merged. They are facts, not proposals.

- **One engine implementation.** `measure/multi_target_workflow.py` is the only
  `MultiTargetWorkflow`. The former duplicate in `measure_gui/multi_target.py` (3240 lines)
  is now a 51-line shim. `measure_api` no longer imports `measure_gui` anywhere.
- **One object registry.** `measure/object_specs.py` defines all 10 types with factory,
  required params, defaults, dependency slots and result type. `measure_api/schemas.py`,
  the v1 factories, and the GUI defaults are derived views. Adding a tool type means
  editing one table.
- **Retired APIs.** `TargetInstance`, `add_target`, `_targets`, `RotatedTemplate`,
  `detect_all`, `_transform_point_result` / `_line` / `_circle` no longer exist. Dead files
  removed: root-level `project_manager.py`, `test_project_manager.py`,
  `tests/test_measure_multi_target.py`, `tests/test_multi_target_workflow.py`. Verified:
  nothing in either generation references them after the merge.
- **v2 must not import GUI or legacy layers — enforced by a test.**
  `tests/v2/test_deployment.py` injects a `BlockGUI` meta-path finder that raises
  `ImportError('V2 imported GUI/legacy dependency: ...')` for any of `measure_gui`,
  `PyQt5`, `PyQt6`, `PySide6`, `tkinter`, `flask`, `waitress`, `matplotlib`. So v2's
  independence from the GUI is a tested guarantee, not a convention — adding such an import
  fails CI rather than merely being discouraged. The engine unification keeps this true:
  `grep measure_gui` over both `measure/` and `measure_api/` returns zero imports.
- **`target_id` is 1-based** and matches the GUI result table numbering. It used to start at
  0 while the GUI showed 1.
- **New modules available to both**: `measure/alignment.py` (`AlignmentStrategy`,
  `AlignResult(patch, M_inv)`, `SingleBoxAlignment`, `MultiPointAlignment`),
  `measure/rectify.py` (pure geometry incl. `crop_and_straighten`, `map_point_via_affine`),
  `measure/result_format.py`.
- **His `geometric_fits.py` refactor is numerically equivalent** to the previous inline fitters.
  Field-by-field comparison on a known circle: `radius`, `max_radius`, `min_radius`,
  `ellipticity`, `mean_error`, `max_error`, `num_points`, `max_radius_point`,
  `min_radius_point` all equal, `(col, row)` ordering preserved, plus new backward-compatible
  `residuals` and `rms_error`. Verified with both implementations on identical input.
- **Model asset format is manifest contract 3.0** (see §0). Legacy single-file `.npz` models
  auto-upgrade on open, keeping the original as `workflow.npz.legacy`.

---

## 6. Test-suite health notes

- **`tests/v2/test_operation_lifecycle.py:224`** sets
  `PYTHONPATH=str(Path(__file__).resolve().parents[2])` when spawning its test server,
  which **replaces** the inherited value rather than appending to it. Any deployment where
  dependencies are not already in site-packages (virtualenv-free CI, `pip --target`, Nix,
  container layer caching schemes) makes these tests fail with a confusing
  `ModuleNotFoundError: No module named 'fastapi'` reported from inside the captured
  `server.log`. Appending fixes it —
  `os.pathsep.join([repo, os.environ.get("PYTHONPATH", "")])` — and with that change all
  21 cases in the file pass. Note this differs from `test_deployment.py:56`, which composes
  `[sentinel, repo]` on purpose: that test *wants* an isolated path so its
  `BlockGUI` meta-path finder can prove v2 never imports GUI/legacy modules. Only the
  lifecycle fixture's blanket replacement looks unintentional.
- `tests/test_1d_measure.py::TestHalcon1DMeasure::test_measure_pairs_4` errors for both
  generations because `data/sample/bottleneck_6.bmp` is absent (`data/` is gitignored).
- After merging his v2 rewrite into the unified-engine branch, the failure set is
  byte-identical to his branch alone: zero regressions introduced by the engine
  unification. Verified by running `tests/v2` on both trees and diffing the sorted
  FAILED lists (`comm -13` empty in both directions).

---

## 7. Known open work

- **OBB / YOLO oriented-box localization** is planned as an `AlignmentStrategy` extension in
  `measure/alignment.py` — i.e. a pluggable *first-stage ROI locator* alongside template
  matching, with rectify-then-measure unchanged underneath. This occupies the same layer as
  v2's `/template-matches`. **Coordinate before building** so the two do not diverge again.
  Design constraints agreed so far: dependency injection so `measure/` keeps zero torch
  imports, model identity recorded as path + sha256, strict anchor-failure semantics, no
  mixing OBB and template localization in one model.
- Whether GUI/v1 object types should converge onto v2's atomic operations (§2 of the
  architecture question) remains undecided.
