# 10: Replace old V2 and verify the full upstream workflow

**What to build:** A deployment operator can start the complete replacement at the V2 HTTP prefix
and an upstream integrator can perform the entire teaching, calibration, preview, production run and
investigation workflow using only the new contract.

**Blocked by:** 09: Convert legacy recipes and replay archived runs.

**Status:** needs-triage

**Spec user stories:** 94.

- [ ] Switch V2 startup and routing to the FastAPI replacement, retire old V2
  sessions/publication/shared-memory contracts and obsolete enforcement tests, and require no GUI
  runtime dependency.

- [ ] Compare every required Spec route/capability family against implemented routes, OpenAPI,
  capability discovery and HTTP tests; no advertised capability may be a stub or omitted.

- [ ] Demonstrate upstream-owned reference image, template and calibration creation; unsaved recipe
  preview; supplied-version multi-target execution; evaluation; diagnostic/report export; and
  archived replay in one representative integration scenario.

- [ ] Provide runnable HTTP client examples and deployment guidance for same-node container root
  aliases, permissions, immutable-input retention, output manifests, operation budgets,
  queue/cache/job limits and upstream cleanup ownership.

- [ ] Verify headless imports/startup, real shared-file exchange across differing mount prefixes,
  complete sync/async equivalence, saturation, cancellation, restart and failure outcomes.

- [ ] Run applicable numerical regressions and public integration checks; keep existing GUI/V1
  usable where retained without treating obsolete V2 behavior as compatibility.

- [ ] Record measured accuracy, repeatability, target count, concurrency, latency and memory on
  representative real inputs; deployment acceptance uses explicitly agreed thresholds rather than
  invented performance guarantees.

- [ ] Publish the final supported contract and known applicability limits, with all implementation
  slices complete before declaring replacement readiness.

