# 08: Inspect and export measurement results

**What to build:** An upstream result viewer can display structured observations as selectable
overlays, request raster and diagnostic files, and export CSV/JSON reports from supplied results
even when it does not retain an engine business-result database.

**Blocked by:** 02: Make computations bounded, cancellable and retryable; 07: Validate, preview and
execute complete supplied recipes.

**Status:** needs-triage

**Spec user stories:** 77, 78, 79, 87, 88, 92.

- [ ] Deliver visualization from supplied results with vector overlays as the normal interactive
  output and optional raster overlays written to the shared output location.

- [ ] Render frames/transforms correctly, including sampled projective circle curves and original
  image coordinates where requested; preserve independent upstream visibility/selection control.

- [ ] Deliver shared-file curves, sampling diagnostics and selected edges without forcing large
  arrays into normal JSON responses.

- [ ] Populate retained operation diagnostics and trace responses with actual per-stage timing,
  queue delay, selected samples, numerical quality and structured failures; unavailable metrics
  remain absent/null.

- [ ] Deliver CSV/JSON export from supplied results with correct values, units, target/node
  identities, validity and supplied evaluation outcomes.

- [ ] Every completed artifact descriptor declares kind, format/schema version, path, digest and
  size; multi-file exports are usable only after final manifest commit.

- [ ] Allow visualization/export from supplied results after the original operational record
  expires; service cleanup does not remove upstream-owned completed reports.

- [ ] Verify exported pixels/vector geometry, raster dimensions, CSV/JSON contents, coordinate
  mappings, file metadata, expiration behavior and failure before manifest commit through HTTP.

