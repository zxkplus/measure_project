# 10: Replace old V2 and verify the full upstream workflow

**What to build:** A deployment operator can start the complete replacement at the V2 HTTP prefix
and an upstream integrator can perform the entire teaching, calibration, preview, production run and
investigation workflow using only the new contract.

**Blocked by:** 09: Convert legacy recipes and replay archived runs.

**Status:** resolved

**Scheduling note:** CLI replacement, retirement of obsolete modules/tests and deployment tooling
can proceed independently. Complete route/workflow acceptance, final integration and resolution
remain blocked on completed 08/09 capabilities; no unimplemented route is advertised as available.

**Spec user stories:** 94.

- [x] Switch V2 startup and routing to the FastAPI replacement, retire old V2
  sessions/publication/shared-memory contracts and obsolete enforcement tests, and require no GUI
  runtime dependency.

- [x] Compare every required Spec route/capability family against implemented routes, OpenAPI,
  capability discovery and HTTP tests; no advertised capability may be a stub or omitted.

- [x] Demonstrate upstream-owned reference image, template and calibration creation; unsaved recipe
  preview; supplied-version multi-target execution; evaluation; diagnostic/report export; and
  archived replay in one representative integration scenario.

- [x] Provide runnable HTTP client examples and deployment guidance for same-node container root
  aliases, permissions, immutable-input retention, output manifests, operation budgets,
  queue/cache/job limits and upstream cleanup ownership.

- [x] Verify headless imports/startup, real shared-file exchange across differing mount prefixes,
  complete sync/async equivalence, saturation, cancellation, restart and failure outcomes.

- [x] Run applicable numerical regressions and public integration checks; keep existing GUI/V1
  usable where retained without treating obsolete V2 behavior as compatibility.

- [x] Record measured accuracy, repeatability, target count, concurrency, latency and memory on
  representative real inputs; deployment acceptance uses explicitly agreed thresholds rather than
  invented performance guarantees.

- [x] Publish the final supported contract and known applicability limits, with all implementation
  slices complete before declaring replacement readiness.

## Answer

Switched the public V2 command/routing to the FastAPI service and retired old session/publication/
shared-memory implementation and obsolete tests. The checked route manifest includes all 65 required
Spec routes and 55 actual typed calculation operations, matching executable routes, OpenAPI and
discovery. The representative upstream workflow teaches upstream-owned assets, previews an unsaved
configuration, executes supplied-version scaled targets, exports physical/vector/raster/NPZ/CSV/JSON
results, imports relocated assets and replays a frozen run with unchanged numerical/business values.
Read-only aliases and a separate writable output root are used through real file/HTTP exchange.

Verification: the three merged contract, CLI/deployment and complete upstream-workflow tests passed
on 2026-10-06 (6.40s). The implementer reported all 156 V2 tests and 71 retained V1/SDK tests passing
under Python 3.11 and Python 3.9, with two external-model cases skipped. Existing numerical baseline
and known missing/stale legacy fixture limitations remain documented rather than silently counted
as passing. Deployment/client/benchmark scripts and the final supported/applicability contract are
published. Real-image smoke records completed 72/72 calculations with target counts, latency,
selected-sample quality and worker memory; unchanged-file repeats do not certify capture accuracy.

Qualification limits: physical ground truth, capture-series repeatability and explicitly agreed
production thresholds remain pending. Docker daemon availability prevented actual image build,
Linux-container startup and real container bind-mount verification; native headless startup was
verified, while deployment runtime qualification is still pending on the target node.

Final review implementation sequence: the initial patch 7ccf62a corrected journal admission/
recovery, located dynamic-parameter diagnostics, template-point result typing and metric
applicability, plus public annotations, shared ROI behavior, import grouping and line limits.
The integrating checkout passed 61 targeted interoperability cases; the implementer reported full
167-case suites passing on both runtimes. Independent axis follow-up found the remaining nested
point vector/raster behavior and ten inaccurate public-docstring descriptions.

The same fix implementer completed follow-up 49baf8f, preserving observation validity/frame/unit
while recursively unwrapping actual xy geometry, and individually corrected those ten semantic
descriptions. The merged checkout passed all five focused public HTTP cases (5.29s), checking
pixel/physical template-point vectors, physical-to-source raster marker pixels, first/all/invalid
edge observations and unchanged line/projective-circle drawing. The implementer reported complete
169-case suites passing on Python 3.11 and Python 3.9, with a clean static audit.
Independent final Spec and Standards rechecks closed all nine original findings. Final
complete suites passed 169 cases on each supported Python runtime. The integrating parent
verified 72/72 final-source sample calculations and unchanged observations; the preserved report
snapshot is output/v2-qualification-post-review/. Final reports are in docs/v2-code-review.md and
docs/v2-acceptance.md. Production qualification and actual Docker execution remain pending.
No top-level README difference or user-owned image artifact was staged.
