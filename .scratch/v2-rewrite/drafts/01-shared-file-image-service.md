# 01: Process shared images through the new FastAPI service

**What to build:** An upstream caller can start the new headless service, inspect a committed shared
image, apply preprocessing or a rotated crop, and receive typed metadata, coordinate mappings and
complete output files through HTTP. This first usable slice establishes the file and execution
contracts using real image operations.

**Blocked by:** None (can start immediately).

**Status:** needs-triage

**Spec user stories:** 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 87, 88, 91, 92.

- [ ] Use FastAPI, Pydantic v2 and Uvicorn with a declared compatible Python runtime; generated
  OpenAPI, interactive documentation and capability discovery reflect only implemented typed
  operations.

- [ ] Deliver image inspection, preview, raw/Canny/Sobel/CLAHE/threshold preprocessing, and
  axis-aligned/rotated cropping as independently callable HTTP operations.

- [ ] Accept shared-root aliases and allowed absolute references; reject ambiguous references, paths
  or resolved symlinks outside configured roots, missing/unreadable inputs and expected-digest
  mismatches.

- [ ] Validate actual GRAY8/RGB/BGR8 image dimensions and channels; preserve floating preprocessing
  data as numeric output and distinguish an 8-bit display preview.

- [ ] Return x/y coordinates, explicit units and source/destination frame mappings; public pose
  angles are degrees and ROI extents have unambiguous full/half-length meanings.

- [ ] Run real computation and large input preparation outside the ASGI event loop in bounded
  processes; provide basic synchronous submission and asynchronous submit/query using the same
  calculation chain.

- [ ] Stage outputs on the destination filesystem, atomically commit completed files and commit
  multi-file manifests last; never overwrite inputs or existing completed assets.

- [ ] Provide stable structured validation/file errors, trace identity, and liveness/readiness tied
  to worker and shared-root availability without GUI initialization.

- [ ] Demonstrate HTTP requests against real temporary shared files, correct crop/preprocessing
  output and responsiveness during computation; retain existing known-good behavior while
  introducing the replacement alongside old V2 until final cutover.

