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

## 1. ⚠️ Not a defect: `measure_pairs()` hard-codes two filters on purpose

`measure/measure1D.py` calls the inner detector with both filter arguments fixed:

```python
self.measure_pos(img, sigma, threshold, "all", "all", False, debug)
#                                          ↑transition  ↑select
```

**This looks like a bug and is not one.** It reads as an oversight often enough to be
worth recording here, because the reasoning lives only in commit history.

- `transition="all"` is required. `measure_pos` would otherwise return only same-signed
  peaks, while the pairing loop needs an adjacent *opposite*-sign pair (`amp1>0 and
  amp2<0`, or the reverse). Handing it a single-polarity list makes every pair impossible,
  so `'positive'`/`'negative'` would silently return nothing. That was the original bug;
  fixed deliberately in `7649fb4`.
- `select="all"` is required for the same reason one level up: `first`/`last` must choose
  between whole edge **pairs**, which the select block further down does. Filtering to one
  edge beforehand leaves no pair to select.

The caller's polarity and selection are honoured — just applied to complete pairs rather
than to individual edges. Verified on an 8-band alternating image (7 edges: all=7,
positive=4, negative=3):

```
transition=all      select=all   -> centers=(14.5, 34.5, 54.5)
transition=positive select=all   -> centers=(14.5, 34.5, 54.5)   # positive-sign pairs
transition=negative select=all   -> centers=(24.5, 44.5, 64.5)   # opposite boundaries
transition=all      select=first -> centers=(14.5)
transition=all      select=last  -> centers=(54.5)
```

Consequently **v2's `/api/v2/measurements/edge-pairs` behaves correctly**: it forwards
`polarity` and `selection`, and both take effect in the pairing stage. So does v1/GUI's
`EdgePairObject`. Nothing here needs fixing.

Guarded by `tests/test_1d_measure.py::test_measure_pairs_polarity_and_select_are_pair_level`
plus a negative control that asserts same-signed peaks genuinely cannot be paired — if that
invariant ever changes, the hard-coding becomes reconsiderable and the test says so.

> A previous revision of this document called the above "a confirmed defect affecting both
> generations" and recommended forwarding the parameters. That recommendation was wrong and
> would have re-introduced the bug `7649fb4` fixed. Retained here as a note because the code
> still invites the same misreading.

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

## 7. Planned work: OBB localization (design agreed, implementation paused)

**Decision taken:** define one shared *localization-source* abstraction and let both
generations adopt it — v1/GUI through `AlignmentStrategy`, v2 through `Localization`.
Neither side invents its own second version of the concept. Implementation is on hold
pending coordination with the v2 owner.

### Why a shared abstraction rather than two implementations

v2 already models exactly the two-stage structure this was designed for. From
`measure_api/v2/recipe_models.py`:

```python
class Localization(StrictModel):
    """One template pose search or a fixed reference ROI, never implicit selection."""
    template: Optional[FileReference] = None
    search: TemplateSearch = ...
    fixed_roi: Optional[RotatedROI] = None
    max_targets: int = Field(default=1000, ge=1, le=1000)

    def one_source(self):          # validator enforces exactly one source
        if (self.template is None) == (self.fixed_roi is None):
            raise ValueError("Localization requires exactly one template or fixed_roi")


class TargetAlignment(StrictModel):
    reference_to_patch: CoordinateMapping
    controls: List[ControlTemplate]
    minimum_controls: int = Field(default=2, ge=2)
    robust: bool = True
    max_residual: float = Field(default=3, gt=0)
    estimate_scale: bool = True
    failure_policy: Literal["fail", "rigid_fallback"] = "fail"
```

So `Localization` **is** the pluggable first-stage ROI locator, and `TargetAlignment`
**is** the rectify-then-measure patch frame. Several constraints that were going to be
new here are already enforced there, and more strictly:

| Intended constraint | Already in v2 |
|---|---|
Strict anchor-failure semantics | `"never implicit selection"` + `failure_policy="fail"` default |
No mixing localization sources | `one_source()` validator rejects zero or two sources |
Affine/scale-aware back-mapping | `estimate_scale`, `robust`, `max_residual` |
Rectified-patch measurement frame | `reference_to_patch: CoordinateMapping` |

Adding OBB as a third arm of `Localization` (`fixed_roi` → `template` → `obb_model`) is a
smaller, direction-consistent change than building a parallel mechanism.

### Agreed design constraints (unchanged by the above)

- **Dependency injection.** An `ObbSource` protocol lives in `measure/`; torch / ultralytics
  imports stay outside it. Note v2 has the mirror guarantee already:
  `tests/v2/test_deployment.py` installs a `BlockGUI` meta-path finder that raises for
  `measure_gui`, `PyQt5/6`, `PySide6`, `tkinter`, `flask`, `waitress`, `matplotlib` — so
  engine-layer purity is tested, not conventional.
- **Reproducible model identity**: path + sha256 (+ `imgsz`, `conf`, class names).
- **Explicit top-k target selection recorded at teaching time**, never re-derived per image.
- **Angle semantics come from the model's learned positive direction**, not from either
  side's stored box angle. See §8 for why this matters for the offset-angle case.

### What is still blocked on the conversation

1. Whether OBB lands in v2 only, v1/GUI only, or both behind the shared protocol.
2. Whether GUI/v1 object types converge onto v2's atomic operations (§2 architecture Q).
3. Who owns the artifact format for a taught OBB model — v2 publishes its own artifacts and
   does not read manifest-v3.0 assets; the GUI writes manifest-v3.0 and cannot read his.

## 8. Reference: what the original OBB consumer actually computes

`srv-bottle-3rd/bottles/wlc/src/wlc2d/wlc2d_offsetangle.py` derives its offset angle from
**two oriented boxes**, not from one box's rotation:

```python
result = model.inference(img)[0].obb
idx = np.argmax(result.conf[result.cls == 0])          # cls 0 = "oritent"; top-1
xywhr_box = result.xywhr[result.cls == 0][idx]         # anchor box centre
# cls 1 = "bottle_center" supplies the second centre (cal_midline_mode dependent)
angle_cal = math.degrees(math.atan2(-(y_obb - cy_cam), x_obb - cx_cam))
angle_cal += initial_left_angle if camera_pos == 'left' else initial_right_angle
angle_cal = (angle_cal + 360) % 360
angle_cal *= rotation_flage                            # ±1 direction flip
```

Consequences for any implementation:

- The value is a **bearing between two detected centres**. It is a composed relation over two
  localizations, not an attribute of a single box, and the OBB's own `r` field is unused here.
- `camera_pos`, `initial_left_angle` / `initial_right_angle` and `rotation_flage` are
  **caller context plus calibration constants** — they must not be captured inside a
  detector abstraction.
- Failure is currently collapsed into `offset_angle = 0` with `status = 'error'`. Any new
  implementation should keep "not found" distinguishable from "measured zero degrees".
- Input is converted BGR → gray → `merge([g,g,g])` before inference, i.e. the model expects
  three channels of luma. Reproduce this exactly or scores will silently differ.
- v2 currently has **no OBB/YOLO capability at all** (verified: no matches for
  obb/yolo/ultralytics under `measure_api/v2/`, and no torch/onnx in `pyproject.toml` or
  `requirements.lock`). Whichever side adds it introduces ML dependencies to that layer, so
  the injection rule above is load-bearing.

## 9. 🔴 Known limitation: `rotation_deg` is unreliable for large ROIs

**Read this before using `rotation_deg` as a measurement.** It applies to v1, the GUI and
the engine alike — v2's template search is a separate implementation and is not covered by
these numbers.

### The defect

`TemplatePoint._enumerate_angles_scales()` (`measure/measure_template.py`) discards any
rotated template variant whose bounding box exceeds the image size:

```python
warped = self._rotate_scale_template(angle, scale)
...
wh, ww = warped.shape
if wh > H or ww > W:
    continue          # silent: no log, no counter, no flag in the result
```

Rotating a rectangle grows its axis-aligned bounds, so a ROI that already fills much of the
frame loses **every** non-zero angle candidate. Only `angle=0` survives, `best_rotation_deg`
is therefore always 0, and `TargetResult.rotation_deg` collapses toward the taught box angle
regardless of how the part is actually oriented.

Measured on the shipped models (reference image rotated by a known amount, then measured):

| model | max error | mean | verdict |
|---|---|---|---|
测试 | **26.9°** | 18.3 | search window effectively dead |
瓶身垂直偏差 | 12.0° | 6.4 | reports ~0.000 at every true angle |
瓶口圆度测量 | 11.9° | 6.2 | same |
测试2 | 11.5° | 6.0 | same |
瓶颈直径圆度测量 | 11.1° | 5.3 | same |
瓶全高 / 瓶身直径测量 | 8.0° | 5.0 | reports exactly 0.000 |
测试3 | 1.0° | 0.6 | OK — its ROI is small relative to the frame |

The pattern is size-driven, not content-driven: seven of eight models are affected, and the
one clean case has a small ROI. Concretely for 瓶身直径测量 (ROI 2488×6048 in a 3648×5472
image): `angle=0` → 3648×5138 kept, `angle=5` → 4082×5438 **dropped**, `angle=15` → 4854×5908
**dropped**.

### Why it hides

- **Self-matching cannot detect it.** Feeding the reference image back gives a true delta of
  0°, which sits exactly on a grid point; measured error was only 0.006° across all 30
  targets. Any regression test must rotate the image.
- The match still *succeeds* — score falls off (1.000 → 0.529) but stays above most
  thresholds, so nothing looks wrong downstream.
- `_refine_1d_subpixel()` runs and appears to refine, but with a single surviving candidate
  there is no parabola to fit.

A secondary issue exists in the multi-target path: `_refine_1d_subpixel()` builds its
parabola from `max(score)` over **all peaks at that angle**, not the peak under evaluation,
so per-match angles get pulled toward a global average (~2° effect). Fixing that alone does
not address the 12–27° errors above.

### Consequences for current work

- Do not present `rotation_deg` as an orientation measurement for large-ROI models.
- A pure-localisation task built on template matching inherits this: position is fine, angle
  is not. Search-free modes (`none`, `fixed_roi`) report `score=None` and angle 0 honestly
  rather than fabricating confidence — see §10.
- OBB-based localisation would sidestep the quantised sweep entirely (model output is
  continuous), which raises the value of landing it.

Not fixed as of this writing: decided to ship the pose plumbing first. Preferred repair is
to crop/pad instead of discarding oversized rotated templates, with a warning plus a
`meta.angle_search_degraded` flag when candidates are dropped.

### A second, unrelated defect in the same field — fixed

Do not conflate the two. Until recently the engine derived the absolute angle as
`box_angle - reported` for *every* locator, but the two locator families encode "rotation
from the taught pose" with opposite signs: a template matcher reports `absolute - taught`
while `ObbAlignment.detect()` reports `taught - absolute` (it formats a detector's absolute
angle into the same field). Subtraction was therefore right for OBB and wrong for template
matching, which reported the negated deviation: exactly correct at the taught pose — the only
case self-matching and unrotated inspection images exercise — and off by *twice* the
deviation once the part actually rotated. Measured on a controlled part: 5 degrees of real
rotation reported as -5 (10 degrees of error), 12 as -10.25 (22 degrees).

The conversion now lives with the strategy (`AlignmentStrategy.absolute_rotation`, overridden
by `ObbAlignment`), and `tests/test_absolute_rotation.py` checks reported angles against
ground truth at three taught angles and five rotations each.

The same sign confusion had a second, worse consequence: the *cropping* angle was derived
with the same shared expression (`box_angle - rotation`), so a template model cropped a
rotated part at **twice** its rotation — a part rotated 14 degrees produced a 29 degree patch.
Tools then ran on a tilted frame: a line taught horizontal followed the part's tilt, and a
scan line crossed the wrong edges. It looks subtle because the patch is small and self-similar,
and because at the taught pose the error vanishes. `AlignmentStrategy.crop_angle` now derives
the crop angle from `absolute_rotation` (both families), so the patch frame and the reported
pose cannot disagree; OBB output is unchanged because it already had the inverse sign built in.

Re-measuring the table above after the fix: the numbers are *unchanged* for seven of the
eight models, which is the expected outcome — their angle search returns 0 for every
candidate, so both sign conventions collapse to the taught box angle and the residual error
here is the discarded-candidate defect alone. Where the search does work (测试3) the reported
angle now follows the part within ~1 degree instead of mirroring it.

## 10. Stage 1 can now be skipped (engine modes `none` and `fixed_roi`)

The engine gained two alignment modes that run no template search, mirroring v2's
`Localization.fixed_roi`:

| mode | patch | mapping | score |
|---|---|---|---|
`none` | the inspection image itself | identity | `None` |
`fixed_roi` | taught ROI cropped and deskewed | affine from `crop_and_straighten` | `None` |

Contract notes for callers:

- **`TargetResult.score` is `Optional[float]`.** Search-free modes report `None`, meaning
  "nothing was correlated" — not zero confidence and not perfect confidence. It serialises
  to JSON `null`; `NaN` was rejected because `json.dumps(allow_nan=False)` raises, which
  would break every API response. Any code formatting or thresholding `score` must handle
  `None` explicitly (`None < t` raises in Python 3).
- **Skipping stage 1 removes detection, not measurement.** With no locator the model returns
  its stored ROI centre for *any* image, including a blank one. If the task needs to know
  whether the part is present, keep a locator (template or OBB); these modes only drop the
  search step.
- **`measurements` may legitimately be empty.** A model with no tools is a supported
  configuration for position-and-angle tasks; `TargetResult.is_localization_only` states it
  explicitly instead of leaving callers to infer it from an empty dict. See
  `TargetResult.valid`'s docstring for how validity is decided in that case.
- **Use `MultiTargetWorkflow.is_taught` to gate save/execute**, never
  `workflow._alignment.template_point is None` — the latter wrongly rejects both new modes
  and was the cause of three GUI refusals.
