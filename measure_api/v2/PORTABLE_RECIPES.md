# Portable recipes, legacy conversion and frozen replay

These calculations own no registry, project session, publication history or latest asset version.
All routes are under `/api/v2`; scheduling/error envelopes are identical to other calculations.
The upstream retains completed files, recipes, asset versions and frozen business records.

## Packages and relocation

`POST /recipe-exports` accepts `recipe` or `recipe_file`, required `output.directory`, and
`packaging: "embedded"` (default) or `"references"`. It uses the same transitive preparation and
full graph validation as `/recipe-validations`. Invalid configuration publishes no completed set.

Embedded output is `recipe.zip`, kind `recipe_package`, with a UTF-8 `manifest.json` and flat
`assets/asset-N.ext` members. Package format `measure-recipe-package` schema `1.0` is separate from
recipe/template/calibration schema versions. The manifest contains the normalized recipe, original
configuration hash and declared asset identities, formats, schema versions, sizes and SHA-256s.
Recipe file references use logical `{root: "package", path: "asset-N", sha256: "..."}` values.
Actual asset bytes retain their digests, including numeric templates and normalized calibration
values. Original calibration/recipe source files remain declared provenance assets.

Reference packaging produces standalone `recipe-package.json` with the same logical recipe and
explicit original shared references, without copying asset bytes.

`POST /recipe-imports` accepts `package: FileReference` and required `output.directory`. Embedded
imports validate private copies using the complete recipe preparer, relocate generated references,
then jointly commit the assets and `recipe.json`. The returned `resolved_recipe` executes directly
on the new node; `configuration_sha256` covers its actual relocated paths and original digests.
`source_package` and `source_configuration_sha256` preserve the original identities separately.
Relocated references retain explicit upstream `provenance` identities and asset versions; import
does not invent, replace or infer them from a new pathname.
Template metadata's original teaching source remains historical provenance; content digest equality
with the relocated recipe reference establishes teaching applicability.

Reference imports resolve configured root aliases, optional `root_bindings` (`old-alias` to
configured `new-alias`) and optional `asset_bindings` (`asset-N` to an explicit shared reference).
Bindings must retain the declared SHA-256. Absolute historical source paths require explicit asset
bindings. Every linked input is read, verified and included in preparation before admission and
idempotency; no old machine pathname is guessed or registered.

```json
{
  "package": {"root": "exchange", "path": "supplied/recipe-package.json"},
  "root_bindings": {"old-machine": "exchange"},
  "output": {"directory": {"root": "exchange", "path": "imports"}}
}
```

Archives are read in memory without extraction. Duplicate, undeclared, traversing, encrypted,
symlink and excessive members fail. Expanded bytes and NPY declared storage are bounded before
array allocation. JSON must be finite; image payloads must decode as supported 8-bit pixels; numeric
archives permit bounded NPY 1.0/2.0 data and Unicode metadata, never object arrays or pickle.
Wrong checksums, unsupported payloads and invalid recipes publish no parent completion manifest.

## One-way legacy conversion

`POST /legacy-project-conversions` accepts `project: FileReference`, explicit upstream
`recipe_id`/`recipe_version`, required `output`, optional `reference` and optional calibration
`applicability`. Supported inputs are:

- Current GUI `project.json` version `3`, whose `template_view.tools` contains complete JSON tools.
- Core `measure.multi_target_workflow` NPZ version `1`, with numeric `template_image` and
  `edge_template`, scalar Unicode JSON `template_params`, `preprocessor_json` and `measure_defs`.

Reference teaching uses the explicit supplied image or the project-local `reference.png`. Old
absolute `reference_image_path` values are never automatically opened. Core NPZ conversion rebuilds
portable templates from this verified reference and reports the changed matching support contract.
It does not unpickle or reuse a GUI loader. GUI/core workflow object-metadata NPZ requires a safe
JSON manifest and returns `pickle_metadata_requires_safe_manifest` as a blocking diagnostic.

Supported tools are `EdgePoint`, `EdgePair`, `FitLine`, `FitCircle`, `TemplateMatchPoint`,
`TwoPointsLine`, `TwoPointsDistance`, `PointLineDistance`, `TwoLinesAngle` and
`PointCircleDistance`. Node IDs derive from unique type/label identities. Compositions use explicit
typed ports; `pair_A` and `pair_B` become `edge_a` and `edge_b`. Forward references resolve before
topological execution. Label/display order never selects a provider. Duplicate/ambiguous labels,
unknown types/parameters and malformed content block readiness.

Rows/columns become x/y; radians become clockwise degrees. Line search/averaging widths preserve
the GUI's documented swapped dimensions; circle radial search widths remain radial. GUI tools are
taught in the straightened patch. The converter records the actual legacy source-to-patch matrix,
including legacy `width/2`, `height/2` center conventions, rather than silently shifting tools by
half a pixel. The rectangular portable template's absolute angular search includes the taught ROI
angle. Raw/Canny/Sobel/CLAHE/threshold presets and supported serialized core parameters are explicit.

Single-box rigid alignment is supported. Legacy multi-point affine alignment returns a blocking
diagnostic because its historical correction composition is not safely equivalent. Scalar mm
calibration requires the positive original square size and explicit camera/setup/frame/image-size
applicability; no camera identity, board spacing evidence or physical scale is invented. UI layout,
zoom, visibility and selection have informational diagnostics; upstream owns those preferences.
Matching overlap/support/pyramid differences are visible warnings requiring upstream migration
validation on representative images.

`valid: false` returns structured `{field_path, severity, code, message}` diagnostics, no ready
recipe and no completed artifacts. `valid: true` returns the validated `resolved_recipe`, assets,
configuration hash, execution order and a jointly committed template/recipe set.

## Freeze and replay

Every new `/runs` and preview result records `runtime_identity`: actual algorithm version,
implementation source digest, Python implementation/version/architecture, imported NumPy/OpenCV/
SciPy versions, OpenCV binary build digest and installed service distribution versions.
`/capabilities` uses the same helper. Decoded file caches use its actual OpenCV/NumPy decoder
identity, shared algorithm version and `IMREAD_UNCHANGED` flags; runtime upgrades cannot reuse
pixels prepared by a different codec in the same state directory.
Older supplied results without producer identity remain unknown; the service never labels them
with its current version. The implementation digest covers shipped `measure/**/*.py` and
`measure_api/v2/**/*.py`; it excludes checkout paths, operational state and documentation.

To freeze a complete run, call `/recipe-exports` with its `resolved_recipe` and either
`frozen_run: RunResult` or `frozen_run_file: FileReference` containing the pure typed result JSON
from `/runs` or `/result-exports`. The export jointly publishes `frozen.json`, kind `frozen_run`,
format `measure-frozen-run`, schema `1.0`. Partial preview results cannot freeze. The supplied run's
configuration hash must agree with the prepared export recipe, and every original replay input
requires its recorded expected digest. A frozen manifest includes the complete prepared recipe,
input image, paired sources, transitive assets, runtime identity, numerical/business expected
record and explicit absolute tolerance (default `1e-9`); prior IDs may be retained as provenance.

```json
{
  "frozen_file": {"root": "exchange", "path": "archives/frozen.json", "sha256": "..."},
  "output": {"directory": {"root": "exchange", "path": "replays"}}
}
```

`POST /replays` accepts exactly one `frozen` inline manifest or `frozen_file`, plus required new
`output`. It verifies versions, every original expected content digest and the resolved configuration
hash before admission, then calls the same recipe executor and atomic chain used by `/runs`.
Historical operation lookup is never required; archived replay works after the prior operation
returns `410`. New operation identity and isolated output namespaces preserve original assets.
Old idempotency controls are absent from the frozen format; replay scheduling is a new request.

Compatibility policy is exact identity equality, including implementation source digest and runtime
dependencies. No accuracy guarantee is inferred across versions or architectures. Missing files
return `file.not_found`; changed bytes return `file.digest_mismatch`; configuration tampering returns
`replay.configuration_mismatch`; incompatible runtime returns `replay.incompatible_version` with
both identities. Invalid frozen schema returns `replay.incompatible_manifest`. These failures do
not create completed replay outputs or infer replacement files.

Response `result.run` is the complete new `RunResult`. `equivalent` and bounded `differences`
compare stable ordered node IDs, numerical geometry/values, frames, units, validity and conformity
with the manifest's tolerance, excluding operation/trace/target IDs, timings and output namespaces.
This comparison reports supported synthetic numerical reproducibility; certified production
accuracy still requires representative optics, cameras and agreed physical ground truth.

## Acceptance evidence

`tests/v2/test_portable_recipes.py` uses public HTTP and real shared files/processes. It exercises
image/numeric-template/calibration relocation, explicit root binding, unsafe archives/pickle/NPY
storage, a 30-degree GUI ROI with forward virtual endpoints and 20px measurement, safe core NPZ,
unsupported compatibility diagnostics, observed version identity, replay after historical expiry,
numerical equivalence and preservation of source bytes. Result export interoperability is checked
through pure exported JSON, frozen export, replay and subsequent visualization. A focused public
shared-file boundary regression replaces only the external OpenCV version/decoder to verify an
upgrade recomputes pixels while original encoded bytes retain their identity.

At the completed Ticket 09 integration gate, all 152 V2 tests pass on Python 3.9. The Python 3.11
suite passed all 151 tests before the decoder follow-up, whose original HTTP cache acceptance and
external-codec regression also pass. After the final explicit asset-provenance relocation regression,
all 29 portable HTTP cases pass again on both Python 3.9 and 3.11, alongside the codec boundary check.
