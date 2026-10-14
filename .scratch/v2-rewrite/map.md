# V2 rewrite implementation

## Notes

Integration branch: feat/v2-rewrite-integration.
Source: the supplied V2 rewrite specification.
The implement-spec instruction approves the preceding ten-ticket task graph.
The user confirmed public FastAPI HTTP tests as the primary seam, with existing
numerical algorithm tests retained for regression.

## Decisions-so-far

Ticket 01 is resolved: the image-service slice passed all 9 merged public HTTP tests.
The public family extension seam is the typed Operation registry, CalculationContext and
CalculationRequest/OperationResponse models. SharedFiles verifies typed input references and
ArtifactWriter commits isolated artifact sets with a manifest last. Family modules expose their
OPERATIONS without editing central HTTP routing.

Ticket 05 is resolved: calibration and image-service acceptance passed 20 merged public HTTP
tests. The implementation also passed 82 tests including pixel/camera/stereo numerical regressions.
Typed calibration asset registration supports additional plane/stereo kinds and generic utilities.

Tickets 03 and 04 are resolved: all 53 merged public HTTP tests passed. Independent batches
exercise actual template-point assets and nested digest verification, completing the deferred
ticket 04 acceptance. Numeric profile batches commit one parent manifest. The line-coordinate
regression returns approximately 40.5 pixels. The implementer reported 21 atomic tests passing on
both supported Python runtimes and 140 broad regressions passing with the absent fixture excluded.

Ticket 06 is resolved: all 64 merged public HTTP tests passed, including generic calibration
validation and strong plane/stereo asset models alongside the previous capability families.
The implementer reported 126 passing tests including numerical regressions and 22 passing
calibration tests under Python 3.9.

Ticket 02 is resolved: all 80 merged public HTTP tests passed, including real server and worker
interruption, publication races, restart, retries, retention, cache and capacity/memory budgets.
The implementer reported 25 lifecycle/image tests passing under Python 3.9. Operation.prepare
and context.stage now provide the shared transitive-input and temporary-output seams.

Ticket 07 is resolved: all 105 merged public HTTP tests passed before the additional file-snapshot
fix; the parent independently verified those 105 under Python 3.9.6 (87.20s). After the fix, all
26 merged recipe tests passed, including restoring the retained stitched canvas from result.json.
The implementer verified both updated stitching variants under Python 3.9. Recipe execution uses typed
preparation/transitive digests, deterministic dependency closure, fresh target outcomes and actual
sample mapping before pixel/physical fitting. Tests verify 24px/2.4mm normalization and same-run
control isolation. The implementer reported all 25 recipe tests passed on Python 3.11 and 3.9.
Only a parent final manifest can mark the complete run.

Ticket 08 is resolved: all 123 merged public HTTP tests passed (82.75s). Typed supplied-result
rendering/export, complete shared diagnostics, expiry-independent reports and final-manifest
cancellation use the existing execution contract. The implementer reported the same full suite
passing on Python 3.11 and Python 3.9, with 33 result/lifecycle timing checks on both runtimes.
HTTP and spawned-worker deadlines now share Unix CLOCK_MONOTONIC, fixing the reproduced
Python 3.9/macOS process-local monotonic epoch difference.
The public V2 CLI/routing now uses FastAPI; obsolete V2 startup/protocol modules and tests are retired.

Ticket 09 is resolved: all 153 merged public HTTP/boundary tests passed (102.48s), including
portable image/template/calibration relocation with explicit provenance, safe legacy conversion,
expiry-independent frozen replay and result-export interoperability. Actual producer/runtime and
decoder identities are shared across capabilities, run results, frozen records and cache keys.
The implementer reported final 29 portable HTTP cases passing on both Python 3.11 and 3.9,
with earlier full Python 3.9 acceptance passing 152 cases before the final provenance case.

Ticket 10 is resolved: three merged public contract/deployment/upstream-workflow tests passed
(6.40s), with all 65 Spec routes and 55 executable typed calculations audited. The implementer
reported 156 V2 and 71 V1/SDK passing tests on Python 3.11 and 3.9, with two external-model skips.
The documented real-image workload completed 72/72 calculations; qualification thresholds and
physical capture truth remain pending, as does actual Docker runtime/bind-mount verification.

All ten implementation tickets and both updates of the single review-fix branch are integrated.
Initial 7ccf62a addressed the four Spec and five Standards findings and passed 61 merged targeted
cases; complete initial-patch suites passed 167 cases on both runtimes per the implementer.
Independent follow-up found remaining nested point vector/raster geometry and ten public-docstring
semantic inaccuracies. Follow-up 49baf8f corrected those paths and descriptions; the merged checkout
passed all five focused point/edge/line/projective-circle HTTP cases (5.29s). The implementer
reported complete follow-up suites passing 169 cases under both Python 3.11 and Python 3.9 and
clean public-boundary/line-length audits.

Independent final Spec and Standards rechecks closed all four Spec and five Standards
findings. Final complete suites passed 169 cases on Python 3.11 and 3.9. The integrating parent
verified 72/72 final-source sample calculations with identical observations to the earlier
snapshot. Final review and workload evidence are recorded in docs/v2-code-review.md and
docs/v2-acceptance.md. Production physical/capture truth, qualification thresholds and actual
Docker runtime/mount verification remain pending.

## Fog

Real-camera production thresholds require representative upstream fixtures.
The legacy numerical baseline contains one case requiring the absent bottleneck_6.bmp sample;
179 other baseline tests passed. This missing fixture predates the replacement implementation.

Ticket 02 is complete; durable operational state is service-owned and upstream artifacts remain upstream-owned.

Ticket 06 is complete; its plane/stereo asset variants extend the shared registry.

Independent integration validation: all 53 public HTTP tests passed under Python 3.9.6
(38.64s) after atomic/template/calibration integration.

Independent integration validation after ticket 02: all 80 public HTTP tests passed under
Python 3.9.6 (68.60s), including bounded queued execution and all capability families.

Broader Python 3.9 numerical/legacy validation: 219 passed, one missing-fixture case deselected.
An additional stale GUI-facing test file, tests/test_multi_target_workflow.py, reports
5 failures and 13 setup errors because it expects an earlier constructor contract.
The same 5 failures and 13 errors were reproduced on untouched baseline 3b46f24 under
Python 3.11 in a temporary checkout; that file and its GUI implementation are unchanged.

Legacy SDK/V1 regression: 71 passed and 2 external-model integration cases skipped
under both Python 3.11 and Python 3.9. The standalone test command overrides addopts because the
--headless option is registered only by the separate tests/conftest.py.

Locked-only final deployment check: the FastAPI factory and all 55 calculation contracts
import with only requirements.lock installed, without measure_gui, PyQt, Flask or matplotlib.
The real CLI/HTTP/shared-path checks and 72 final-source sample calculations used this runtime.

Retained stitched result-file snapshots now preserve the same measurement_image and stage
provenance as HTTP; a public request can inspect and measure the restored canvas after completion.
