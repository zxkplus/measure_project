# Localisation (stage 1) — finding the part, with or without measuring it

Everything in this file applies to the shared `measure/` engine that both API generations and
the GUI sit on. Read it before answering "how do I get the part's pose / how do I use an OBB
model", because the two HTTP generations expose different slices of it.

## The two stages

```
stage 1  locate    → where is the part? (centre, rotation, scale, score, patch frame)
stage 2  measure   → what do its features measure? (tools defined in the patch frame)
```

Stage 1 is pluggable (`AlignmentStrategy`, `measure/alignment.py`). Stage 2 is a list of
measurement tools. **A model with no tools is a supported configuration, not a degenerate
one**: the pose is then the entire deliverable. That is "localisation only".

## The pose is a first-class result

```python
targets = workflow.measure(image)
for t in targets:
    t.pose            # {'center_row', 'center_col', 'angle_deg', 'scale', 'score'}
    t.is_localization_only   # True when no tools were defined
    t.valid           # locator matched (and every tool was valid, if any exist)
```

- `pose` is a **view** over the same fields as `to_dict()["pose"]` — never a copy, so it
  cannot drift.
- `center_row` / `center_col` are in inspection-image pixels.
- `angle_deg` is the target's **absolute orientation**, visually counter-clockwise positive,
  the same sense as `measure.rectify.compute_rotated_box_corners`. It is directly usable to
  draw the box — no extra sign change.
- `score` is `None` for the search-free modes (`none`, `fixed_roi`). It is *not* `0.0` and
  not a fabricated `1.0`: nothing was correlated, so there is nothing to report. Any
  comparison against a threshold must handle `None` explicitly.
- `valid` with no tools means "the locator matched". It does **not** mean "the part is
  present" for the `none` mode — that mode returns its stored frame for any image, including
  an empty one.

## Alignment modes

| `mode_name` | How it locates | Pose source | API/GUI |
|---|---|---|---|
| `single_box` | NCC template search over a rotated box | template match | v1, GUI |
| `multi_point` | template match + control-point affine refinement | template match | v1, GUI |
| `obb` | injected detector, one oriented box per instance | detector | **GUI only** |
| `fixed_roi` | nothing — a fixed rotated ROI | taught angle | engine, v2 (`Localization.fixed_roi`) |
| `none` | nothing — the whole image is the frame | taught angle | engine, GUI (display) |

The v1 API only ever creates `single_box` (`set_template`). The GUI can create `single_box`,
`multi_point` and `obb`. `fixed_roi` / `none` can be *loaded* by the GUI but not selected
there.

## Angles: do not re-derive them yourself

Three conventions coexist, and two locator families report their delta with opposite signs.
Both traps are silent — they vanish at the taught pose (rotation 0), which is what
self-matching and unrotated test images exercise.

```
manifest / compute_rotated_box_corners   visually CCW     (+)
crop_and_straighten                      maths CCW on (x, y) = (col, row)
ImageCanvas._roi_angle                   visually CW
```

Use the strategy's own conversions instead of writing the arithmetic again:

```python
strategy.absolute_rotation(reported)   # match's reported rotation → absolute orientation
strategy.crop_angle(reported)          # angle to hand to crop_and_straighten
```

- A template matcher reports `absolute - taught`; `ObbAlignment.detect()` reports
  `taught - absolute`. Same field, opposite sign. That is why the conversion lives on the
  strategy.
- Getting either one wrong produces an error of **twice** the deviation, mirrored, with a
  correct-looking value at the taught pose. `tests/test_absolute_rotation.py` pins both the
  reported angle and the straightness of the cropped patch.

## OBB: detector-backed localisation

A YOLO-OBB detector supplies stage 1; its oriented box becomes the ROI. Everything about
stage 2 (tools, patch frame, results) is unchanged.

### Engine usage

```python
import cv2

from measure.alignment import ObbAlignment
from measure.multi_target_workflow import MultiTargetWorkflow
from measure_obb.ultralytics_source import UltralyticsObbSource

reference = cv2.imread("reference.png", cv2.IMREAD_GRAYSCALE)
source = UltralyticsObbSource("weights/yolov8n-obb.pt", conf=0.5, imgsz=1024)

detections = source.detect(reference)          # self-check BEFORE teaching
if not detections:
    raise SystemExit("detector finds nothing on the reference image")
anchor = max(detections, key=lambda o: o.score)  # or let a user pick a box

strategy = ObbAlignment()
workflow = MultiTargetWorkflow()
workflow._alignment = strategy
strategy.teach(
    reference,
    source=source,
    anchor_class_id=anchor.class_id,
    anchor_observation=anchor,     # the box that fixes the patch frame, not just the class
    min_score=0.5,
    model_ref=source.serialize(),  # path + sha256 + conf + imgsz + class names
)

for target in workflow.measure(inspection_image):
    print(target.pose)             # add no tools → localisation only
workflow.save("models/my_obb_model")
```

- `measure/` never imports torch or ultralytics: the detector is injected through the
  `ObbSource` protocol (`measure/obb_source.py`), and `measure_obb/` is the adapter.
- **The anchor must be a detection, not a class.** On a sample with 166 ships, "ship" does
  not say which box sets the patch frame and the anchor pose. Pass `anchor_observation`
  (the GUI lists every detection and preselects the highest-scoring one).
- Teaching refuses a reference image on which the detector finds nothing — otherwise every
  later image would return zero targets with no hint that setup was the cause.
- The **detector is not persisted**. The manifest records its identity (`model_ref`:
  path, sha256, conf, imgsz, task, class names); the caller re-attaches a live source. The
  GUI does that automatically on project open and greys out Execute when the weights are
  missing.

### Constraints

- `ultralytics` must be installed (in this repo: the `y8_env` conda env). The GUI catches the
  missing import and shows install instructions instead of crashing.
- The patch frame is the model's **fixed nominal size**, taken from the anchor. Every
  instance is cropped to that same size, so tools keep their taught meaning; a detector's
  own per-instance `box_size` is recorded but deliberately not used for the crop. Consequence:
  on parts whose proportions vary a lot, tools placed in patch coordinates can miss — that is
  stage 2, not localisation.
- `ObbObservation` carries `class_id`, `class_name`, `score`, `corners`; `from_corners`
  rejects a non-rectangle (tolerance `1e-3`, needed because float32 detector corners miss
  orthogonality by ~1e-6).

### GUI flow

Set 对齐方式 (alignment mode) to **OBB 模型** → browse for the weights → click
`载入模型并自检` (loads the model and runs one self-check detection on the reference image,
listing every detection as a selectable anchor, highest score preselected) → optionally pick a
different anchor → leave the tool list empty for localisation-only → `执行` (Execute).

## Per-generation access

### v1 (`/api/session/*`, documented in this skill)

v1 can only build template models (`set_template`), so localisation-only is expressed as
"template + **zero** measurement tools":

- phase stays `template_ready`, and `POST /measure` accepts it;
- each target has `measurements == {}`, and `valid` is decided by the match alone;
- the response target has flat `row`, `col`, `rotation_deg`, `scale`, `score` — there is
  **no `pose` key** and no `is_localization_only` flag. Callers infer "localisation only"
  from `measurements == {}`.

**v1 has no OBB entry point.** There is no model-path parameter and no way to inject a
detector.

### v2 (`/api/v2/*`, NOT documented here — owned separately)

- `POST /api/v2/template-matches` is the standalone localisation operation: it returns
  `matches[]` with `center`, `anchor`, `angle_degrees`, `scale`, `score`, `mapping`,
  `support_mapping`. Use it when the answer *is* the pose.
- A recipe run also localises: `Localization` (`template` XOR `fixed_roi`) plus an optional
  `TargetAlignment`, and each `targets[]` entry carries `localization.match` and `mapping`.
  But `Recipe.nodes` requires **at least one node**, so a recipe cannot be "localisation
  only" — use `template-matches` for that, or accept one dummy node.
- v2's `angle_degrees` is the **search angle applied to the template** (a delta from the
  reference pose), like v1's matcher. If you need an absolute orientation, derive it from
  `mapping` rather than assuming a sign.
- **v2 has no OBB capability**: no route, no `Localization` arm, no torch dependency. Making
  v2 accept an OBB source means adding a third arm to `Localization` and a dependency-
  injected source — the design discussion is recorded in
  [v1-v2-divergence.md](v1-v2-divergence.md) §7.

### Summary

| | localisation-only | OBB |
|---|---|---|
| engine (`measure/`) | yes | yes |
| GUI | yes (teach an anchor, add no tools) | yes |
| v1 API | yes (template, zero tools; no `pose` key) | **no** |
| v2 API | via `/template-matches` (not as a recipe) | **no** |

## Pitfalls

- **Do not re-derive the angle.** Any place that computes `box ± reported` by hand will be
  wrong for one of the two locator families. Call `absolute_rotation()` / `crop_angle()`.
- **A rotated part is the only test that proves the pose path.** Self-matching reports zero
  rotation and hides both sign bugs. Rotate an image by a known amount and check the output.
- **`score=None` is not a failure.** It means no search ran. `None < threshold` raises.
- **Do not lower a tool's edge threshold to silence `Only N edge points found`.** That warning
  means the caliper found fewer than 2 edges above threshold; lowering the threshold makes it
  fit interior texture and returns a confidently wrong number.
- **The detector is an injected capability, not part of the asset.** Reopening an OBB model
  without re-attaching a source leaves Execute disabled — that is deliberate; without it the
  model would report zero targets and read as "no part on the image".
