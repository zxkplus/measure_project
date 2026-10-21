# 01: Process shared images through the new FastAPI service

**What to build:** An upstream caller can start the new headless service, inspect a committed shared
image, apply preprocessing or a rotated crop, and receive typed metadata, coordinate mappings and
complete output files through HTTP. This first usable slice establishes the file and execution
contracts using real image operations.

**Blocked by:** None (can start immediately).

**Status:** resolved

**Spec user stories:** 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 87, 88, 91, 92.

- [x] Use FastAPI, Pydantic v2 and Uvicorn with a declared compatible Python runtime; generated
  OpenAPI, interactive documentation and capability discovery reflect only implemented typed
  operations.

- [x] Deliver image inspection, preview, raw/Canny/Sobel/CLAHE/threshold preprocessing, and
  axis-aligned/rotated cropping as independently callable HTTP operations.

- [x] Accept shared-root aliases and allowed absolute references; reject ambiguous references, paths
  or resolved symlinks outside configured roots, missing/unreadable inputs and expected-digest
  mismatches.

- [x] Validate actual GRAY8/RGB/BGR8 image dimensions and channels; preserve floating preprocessing
  data as numeric output and distinguish an 8-bit display preview.

- [x] Return x/y coordinates, explicit units and source/destination frame mappings; public pose
  angles are degrees and ROI extents have unambiguous full/half-length meanings.

- [x] Run real computation and large input preparation outside the ASGI event loop in bounded
  processes; provide basic synchronous submission and asynchronous submit/query using the same
  calculation chain.

- [x] Stage outputs on the destination filesystem, atomically commit completed files and commit
  multi-file manifests last; never overwrite inputs or existing completed assets.

- [x] Provide stable structured validation/file errors, trace identity, and liveness/readiness tied
  to worker and shared-root availability without GUI initialization.

- [x] Demonstrate HTTP requests against real temporary shared files, correct crop/preprocessing
  output and responsiveness during computation; retain existing known-good behavior while
  introducing the replacement alongside old V2 until final cutover.

## Answer

Implemented on the integration branch through the ticket 01 merge. The replacement exposes typed
image inspection, previews, all five preprocessing variants and rotated crops through FastAPI.
Shared references are verified before acceptance and again during execution, while output artifacts
are committed without overwrite and completed by a final manifest. Input hashing and calculations
run in bounded isolated processes; synchronous and asynchronous requests share the execution path.

Verification: the merged integration checkout passed all 9 public HTTP acceptance tests on
2026-10-05. The implementer also verified the application using a real Uvicorn socket. The retained
legacy numerical baseline passed 179 tests; one existing fixture-dependent case requires the absent
bottleneck_6.bmp sample and was recorded as an environmental limitation. No legacy algorithm or
user-owned image artifact was changed by this ticket.

Deadline, cancellation, retention and idempotency semantics are completed by ticket 02; this first
slice establishes their public scheduling fields and basic asynchronous identity/query behavior.
