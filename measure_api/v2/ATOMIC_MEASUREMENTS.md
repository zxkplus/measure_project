# Atomic measurement contracts

All operations use the shared calculation envelope and shared-file controls. Coordinates are
image x/y (x right, y down), clockwise positive angles in degrees, and pixel units for image
observations. Named output ports are discoverable through capabilities and use real result fields.
The implementations are callable through the operation registry without an HTTP or GUI dependency.

## Image probes and fits

- `edge-profiles`: image, probe, sigma and explicit gray/fraction threshold. Return actual gray,
  smoothed, gradient and source sample positions. Optional output commits numeric NPZ curves with
  a final manifest. `inline_samples=false` requires output and suppresses inline arrays.
- `measurements.edge-points`: image and probe plus polarity and first/last/all selection. Return
  signed amplitudes and subpixel positions in the `edges` array. The `point` port is a positioned
  point only when exactly one edge is selected; otherwise it is null.
- `measurements.edge-pairs`: image and probe plus polarity and first/last/all/endpoints selection.
  Detect all boundaries before pairing and selecting. `pairs` is an edge-pair array including
  signed boundary amplitudes. The single-value `edge_a`, `edge_b`, `center` and `width` ports are
  present only when exactly one pair is selected. Boundary and center ports are positioned points;
  width is a scalar observation. `endpoints` chooses the outermost detected boundaries.
- `measurements.lines`: expected start/end xy positions and probe controls. Each probe searches
  perpendicular to the expected line. Nearest/first/last/all selection is explicit. Return `line`,
  selected edge amplitudes and fit diagnostics, including requested/successful probe counts.
- `measurements.circles`: expected center/radius, optional radius limits and clockwise arc bounds.
  Probes search inward. Filter radius limits before nearest/first/last/all selection. Arc end equal
  to start means a full circle; otherwise bounds wrap clockwise. Return `circle`, selected edges,
  actual residuals and `sampled_diameter_spread`.

A probe uses `search_half_length` along its clockwise orientation and `average_half_width`
perpendicular to it. `interpolation` is nearest/linear/cubic; `border_policy` is reject/replicate.
The default threshold is 30 gray derivative units and sigma is 1. Fraction thresholds range from
0 to 1 and are multiplied by 255. Fitting defaults to nearest edge, 20 line probes or 24 circle
probes, search half-length 15 and averaging half-width 5. OpenAPI is authoritative for all bounds.

## Geometry and observation types

Positioned points contain `valid`, `reason`, `frame`, `unit` and `point: {x,y}`. Scalars contain the
same observation metadata plus `value`. Lines contain normalized `a*x+b*y+c=0` coefficients and
optional start/end positions. Circles contain xy center and positive radius. Invalid observations
return null geometry/value rather than a valid zero dimension.

- `geometry.lines_fit` and `geometry.two-point-lines` return a single `line` port with samples and
  fit diagnostics. Supplied-point line fitting uses orthogonal least squares.
- `geometry.circles_fit` returns `circle`, samples, diagnostics and sampled diameter spread.
  Circle fitting uses translation-conditioned algebraic least squares and rejects collinear inputs.
- `geometry.point-distances` takes positioned `point_a`/`point_b`; returns Euclidean distance.
- `geometry.point-line-distances` takes positioned `point` and `line`; returns perpendicular offset.
- `geometry.line-angles` takes `line_a`/`line_b`; returns the smaller unsigned angle in 0..90 degrees.
- `geometry.point-circle-distances` takes positioned `point` and `circle`; returns the shortest
  circumference distance, including points inside the circle.
- `geometry.circle-dimensions` takes circle and optional positioned samples; returns scalar radius,
  diameter and sampled diameter spread ports. Spread needs at least two valid compatible samples.

All geometric composition requires compatible frames and units. A supplied invalid observation
propagates invalidity. RMS comes directly from the selected samples' residuals. Sampled diameter
spread is exactly `2*(max sampled radius-min sampled radius)`; it is not an ellipse or a standardized
roundness measurement. Coordinates from normalized patches must first be mapped back to the
measurement frame before composition; these operations do not invent a calibration.

## Independent batches

`measurement-batches` takes one image and `measurements`, each with a unique `id`, a registered
atomic image-measurement `operation` identifier/path, and `parameters` omitting image. Registered
local template-point operations are supported by the same dispatch. Each child request is resolved
into its real typed model before recursive shared-file verification, including template assets.
The batch returns ordered per-item state, validity, typed result content or structured error.
An independent failure preserves other results; outer validity is false for a partial failure.

Optional batch `output` aggregates all profile numeric files into one operation namespace and one
final parent manifest. Children execute in memory, and the parent commits the payloads once after
all calculations finish. Per-item descriptors reference that same manifest. Child profile output
settings, if supplied, must use the same common directory. Inline curve suppression remains per item.

## Product acceptance

`evaluations.ranges` consumes a scalar observation. Minimum/maximum bounds are inclusive and at
least one bound is required. `evaluations.diameter-variation` calculates max diameter minus minimum
diameter from at least two valid supplied scalars; scalar lengths may come from separate image
frames but must use the declared common unit. `evaluations.centerline-deviation` calculates maximum
center x minus minimum center x from at least two compatible positioned observations.

All three require explicit common limit units (default px), and return `evaluation_status` as pass,
fail or not_evaluated independently from computational completion and observation validity.
`invalid_policy` is not_evaluated (default), fail, error, or ignore. Ignore allows aggregate metrics
from remaining valid samples, but never makes an insufficient set valid. No rounding occurs before
comparison. Upstream remains responsible for business policy and applicable calibrated units.
