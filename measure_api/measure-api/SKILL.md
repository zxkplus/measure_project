---
name: measure-api
description: |
  Industrial measurement REST API for template-based geometric part inspection. 
  Use when a user asks to: measure bottle / container dimensions (diameter, radius, edge distance, angle), 
  inspect industrial parts via machine vision, call a measurement REST API, build a web or desktop 
  application around an industrial vision pipeline, perform NCC template matching measurements, 
  run batch inspection across multiple production images, or create measurement models for 
  production quality control. The API has a two-phase lifecycle: Teaching (build + tune a model)
  and Measurement (run model on production images). All endpoints return JSON.
---

# Measure API

## Scope of this skill

This skill documents the **v1 session-oriented API** under `/api/session/*`. A second,
newer generation exists at **`/api/v2/*`**: a *stateless* FastAPI computation service
organized as atomic operation families (`templates`, `alignment`, `measurements`,
`geometry`, `calibrations`, `plane_stereo`, `recipes`, `results`) with typed port
compatibility, upstream-owned shared-file references, and sync/async operations via
`wait_ms` → 200/202. It has no modeling session and no ROI resource.

v2 is **not** covered here. Its authoritative contract is `docs/v2-openapi.json`
(63 paths, 222 schemas), with field-level docs in `docs/v2-api-reference.md`; see
[generated/v2_surface.md](references/generated/v2_surface.md) for a rendered route index.
Do not mix the two generations — they also write different on-disk formats.

**Before touching shared code, porting a model, or calling the other generation, read
[v1-v2-divergence.md](references/v1-v2-divergence.md).** It records where the two
implementations currently disagree (dependency type rules, template-search defaults,
frame/unit tracking), one confirmed defect in shared edge-pair code that affects both,
which side owns each difference, and what is still undecided.

## Overview

The Measure API is a RESTful service for industrial machine vision measurement. It wraps the shared `measure/` engine (template matching + geometric fitting, tkinter GUI on top) into HTTP endpoints.

**Two-phase lifecycle:**

- **Teaching (Modeling):** Load a reference image → define template ROI → add & tune measurement tools (circles, lines, edges) → optionally compose relationships → save the model.
- **Measurement (Production):** Load a saved model → run batch measurement on inspection images → extract geometric values (radius, distance, angle) with quality metrics.

**Session state machine — enforce call ordering:**

```
CREATED → load_reference → REF_LOADED → set_template → TEMPLATE_READY
→ add_measurement(s) → HAS_MEASUREMENTS → add_composed → HAS_COMPOSED
→ save / load → measure → MEASURED
```

Attempting an operation in the wrong phase returns a 400 error with a descriptive message.

**Traceability:** Every response includes `X-Trace-Id` header. Log entries and persisted call records use the same trace ID for cross-referencing.

## Using This Skill

When a user asks to work with the Measure API:

1. **Identify the phase.** Teaching (building a model from a reference image) or Measurement (running a saved model on production images)?
2. **Determine session scope.** New session or reusing an existing one?
3. **Respect the state machine.** Call endpoints in the correct order — a reference must be loaded before the template can be set, etc.
4. **Test during teaching.** Use `include_visual=true` for immediate visual feedback when tuning parameters.

## Reference Files

| File | When to read |
|---|---|
| [api_reference.md](references/api_reference.md) | When writing v1 API call code — endpoint list, request/response schemas, curl examples |
| [workflow_teaching.md](references/workflow_teaching.md) | When building a measurement model from scratch — detailed steps with parameter guidance |
| [workflow_measurement.md](references/workflow_measurement.md) | When running batch measurement — loading model, interpreting results, error recovery |
| [generated/object_contracts.md](references/generated/object_contracts.md) | **Authoritative object types, params and defaults** for v1 *and* v2 |
| [generated/v2_surface.md](references/generated/v2_surface.md) | v2 route index + schema index, rendered from `docs/v2-openapi.json` |
| [v1-v2-divergence.md](references/v1-v2-divergence.md) | **Read first for cross-generation work** — conflicts, shared defects, undecided ownership |

The two files under `references/generated/` are machine-derived from the code and
pinned by `tests/test_generated_api_docs.py`; they cannot go stale silently. If you
need a parameter default or type name, read those rather than trusting prose. To
regenerate after changing the engine: `python scripts/gen_api_reference.py`.

## Common Pitfalls

- **`object_type` must come from the registry — there are exactly 10.** Primitives: `FitCircle`, `FitLine`, `EdgePoint`, `EdgePair`, `TemplateMatchPoint`. Composed: `TwoPointsDistance`, `TwoPointsLine`, `TwoLinesAngle`, `PointLineDistance`, `PointCircleDistance`. Names like `FitEdge`, `FindCircle`, `FitRectangle`, `FindLine`, `PointToLineDistance` do **not** exist and return 400.
- **Wrong call order**: Must call `load_reference` → `set_template` → `add_measurements` → `measure`. The session phase tracks this — check with `GET /api/session/<sid>`. State-machine violations come back as **400**, not 409.
- **Phase guards are checked *before* argument validation.** If you are in the wrong phase, a bad `object_type` will not be reported — you get the phase error instead. Fix the phase first, then re-check your payload.
- **File paths on server**: All image paths are server-side file paths. The server reads them directly; you cannot upload images through the API.
- **`max_matches` = 0 means unlimited**: If you want single-target mode, set `max_matches=1`.
- **`angle_range_deg` is a half-range**: 30 means ±30°. The four optional search fields (`angle_step`, `coarse_angle_step`, `coarse_fine`, `overlap`) are easy to omit by accident and are persisted with the model — `angle_step` in particular trades speed against orientation accuracy.
- **`target_id` is 1-based**, matching the GUI result table. It is not an index into `targets`.
- **Branch on `measurements[label]["type"]`**, not on key presence. Line results have no length field — derive it from `start_row/col` and `end_row/col`.
- **`include_visual=true` means different things per endpoint.** On `/measurements` and `/composed` it overlays only that one tool; on `/measure` it draws **all** detected targets and all their measurements (a full-scene PNG). Both are base64 PNG — size accordingly in production.
- **`quality` is sparse and partly misleading if taken literally.** `expected_edges` is usually `0` (the fitters don't record `num_measures` in `meta`), which forces `coverage_ratio` to `0.0`; the `rms` key is never actually emitted. Judge fit quality from `result.meta.mean_error` / `max_error` (pixels) and `meta.num_points`, and always read optional keys with `.get()`.
- **Composed dependencies are type-checked, with one deliberate exception.** A `point` slot also accepts a fitted circle (`CircleResult.row/col` alias its centre) — the shipped models rely on that. Anything else mismatches and now returns 400 instead of failing silently at run time. But a dependency may only name an object that actually publishes results: a **valid** `EdgePair` exposes just `<label>_A` / `<label>_B`, never its bare label. Pointing at the bare label is rejected with a message naming both endpoints.
- **A run-time exception truncates every later tool and adds `"_error"`.** The engine wraps the whole per-target tool loop in one `try`, so if tool *N* raises, tools *N+1…* are never evaluated and `targets[].measurements` gains a string-valued `"_error"` key. Do not iterate `measurements` assuming every entry is a result dict — skip `_error` (and any missing label) explicitly. A cleanly *invalid* measurement, by contrast, does appear as `{"type": ..., "label": ..., "valid": false}` with no value keys.
- **`POST /measurements` registers the tool even when the test fails.** The definition is appended regardless of `valid`, so tuning by repeated POSTs silently accumulates tools and eventually hits `Duplicate measurement label`. Iterate with `POST /measurements/test` (no side effects) or `PUT /measurements/<label>`, and only POST once you are happy.
- **Cascade delete**: Deleting a primitive measurement removes all composed measurements that depend on it.
- **v1 storage is not v2 storage**: `save()` writes `config.json` + `template.npz` + `reference.png`. The GUI and the v2 API use a different manifest-v3.0 asset (`manifest.json` + `reference.npy` + `workflow.npz`). The two directories are not interchangeable.
