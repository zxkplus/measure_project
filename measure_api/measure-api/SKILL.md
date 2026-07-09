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

## Overview

The Measure API is a RESTful service for industrial machine vision measurement. It wraps a PyQt-based template matching + geometric fitting engine into HTTP endpoints.

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
| [api_reference.md](references/api_reference.md) | When writing API call code — complete endpoint list, request/response schemas, curl examples |
| [workflow_teaching.md](references/workflow_teaching.md) | When building a measurement model from scratch — detailed steps with parameter guidance |
| [workflow_measurement.md](references/workflow_measurement.md) | When running batch measurement — loading model, interpreting results, error recovery |

## Common Pitfalls

- **Wrong call order**: Must call `load_reference` → `set_template` → `add_measurements` → `measure`. The session phase tracks this — check with `GET /api/session/<sid>`.
- **File paths on server**: All image paths are server-side file paths. The server reads them directly; you cannot upload images through the API.
- **`max_matches` = 0 means unlimited**: If you want single-target mode, set `max_matches=1`.
- **`include_visual=true`**: Only visualizes the first measurement/result. Use for debugging, not production.
- **Cascade delete**: Deleting a primitive measurement removes all composed measurements that depend on it.
