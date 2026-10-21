# 08: Inspect and export measurement results

**What to build:** An upstream result viewer can display structured observations as selectable
overlays, request raster and diagnostic files, and export CSV/JSON reports from supplied results
even when it does not retain an engine business-result database.

**Blocked by:** 02: Make computations bounded, cancellable and retryable; 07: Validate, preview and
execute complete supplied recipes.

**Status:** resolved

**Spec user stories:** 77, 78, 79, 87, 88, 92.

- [x] Deliver visualization from supplied results with vector overlays as the normal interactive
  output and optional raster overlays written to the shared output location.

- [x] Render frames/transforms correctly, including sampled projective circle curves and original
  image coordinates where requested; preserve independent upstream visibility/selection control.

- [x] Deliver shared-file curves, sampling diagnostics and selected edges without forcing large
  arrays into normal JSON responses.

- [x] Populate retained operation diagnostics and trace responses with actual per-stage timing,
  queue delay, selected samples, numerical quality and structured failures; unavailable metrics
  remain absent/null.

- [x] Deliver CSV/JSON export from supplied results with correct values, units, target/node
  identities, validity and supplied evaluation outcomes.

- [x] Every completed artifact descriptor declares kind, format/schema version, path, digest and
  size; multi-file exports are usable only after final manifest commit.

- [x] Allow visualization/export from supplied results after the original operational record
  expires; service cleanup does not remove upstream-owned completed reports.

- [x] Verify exported pixels/vector geometry, raster dimensions, CSV/JSON contents, coordinate
  mappings, file metadata, expiration behavior and failure before manifest commit through HTTP.

## Answer

Implemented visualization and CSV/JSON export from typed supplied atomic, run and batch records or
verified result-file snapshots. Requests validate nested numerical geometry and file descriptors,
pin transitive identities before acceptance, and need no originating operation. Selectable vector
primitives preserve frames and precision; optional raster output uses actual image dimensions,
projective circles use sampled curves and supported nonlinear inverse mapping applies K/D.
Full sampling/curve arrays and previews are independent bounded NPZ/PNG diagnostics, committed
with requested reports under a single final manifest. CSV preserves units, identity, precision,
invalid empty values and supplied evaluation state. Retained diagnostics expose actual queue,
startup, preparation, computation, samples, numerical quality and structured failures.

Verification: all 123 public HTTP tests passed on the merged integration checkout on 2026-10-06
(82.75s). Coverage includes typed result/file restoration, vector/raster mapping, exact numerical
NPZ diagnostics, CSV/JSON content, expiry-independent calculations, artifact ownership and
cancel/timeout before a multi-file manifest. The implementer reported the same 123 tests passing
under Python 3.11 and Python 3.9 and 33 result/lifecycle tests passing while verifying timing.
A reproduced Python 3.9/macOS per-process monotonic-origin difference required the shared Unix
CLOCK_MONOTONIC helper; HTTP budgets and worker commit checks now use that OS epoch consistently.
Measured duration diagnostics remain real process-local elapsed intervals.

Tickets 09 and 10 remain active in parallel. Ticket 09's result interoperability gate is now
complete; final ticket 10 integration remains gated by ticket 09. No top-level README
case-collision difference or user-owned image artifact was staged.
