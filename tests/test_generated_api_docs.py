"""Pin generated API docs against the code they describe.

``scripts/gen_api_reference.py`` renders documentation from ``measure.object_specs``
and the v2 app's own OpenAPI document.  These tests fail when the checked-in output
no longer matches what the code produces — i.e. when someone changed a contract and
left the docs behind, which is exactly how the previous hand-written reference drifted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.argv = [sys.argv[0]]  # keep argparse out of the way on some runners
import scripts.gen_api_reference as gen  # noqa: E402


def _regenerate():
    """Return {filename: expected_text} without touching disk."""
    return gen.render()


class TestGeneratedDocsAreCurrent:
    """The committed generated files must equal what the code yields now."""

    @pytest.mark.parametrize("name", sorted(gen.TARGETS))
    def test_file_on_disk_is_up_to_date(self, name: str) -> None:
        expected = _regenerate()[name]
        path = gen.GENERATED_DIR / name
        assert path.exists(), (
            f"{path.relative_to(ROOT)} is missing. "
            f"Run: python scripts/gen_api_reference.py"
        )
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            first = next(
                (i for i, (a, b) in enumerate(zip(actual.splitlines(),
                                                  expected.splitlines()))
                 if a != b),
                min(len(actual.splitlines()), len(expected.splitlines())),
            )
            pytest.fail(
                f"{path.relative_to(ROOT)} is STALE — the code changed but the "
                f"generated doc did not (first difference at line {first + 1}).\n"
                f"Run: python scripts/gen_api_reference.py\n"
                f"on disk : {actual.splitlines()[first] if first < len(actual.splitlines()) else '<eof>'}\n"
                f"expected: {expected.splitlines()[first] if first < len(expected.splitlines()) else '<eof>'}"
            )

    def test_banner_marks_files_generated(self) -> None:
        for name in gen.TARGETS:
            text = (gen.GENERATED_DIR / name).read_text(encoding="utf-8")
            assert text.startswith("<!-- AUTO-GENERATED"), name
            assert "do not edit" in text.splitlines()[0], name


class TestObjectContractsMatchRegistry:
    """The table must list every registered type and nothing else."""

    def test_every_registered_type_appears(self) -> None:
        from measure.object_specs import OBJECT_SPECS

        text = (gen.GENERATED_DIR / "object_contracts.md").read_text(encoding="utf-8")
        for object_type in OBJECT_SPECS:
            assert f"`{object_type}`" in text, f"{object_type} undocumented"

    def test_no_phantom_types(self) -> None:
        """Names the old hand-written reference invented must not come back."""
        from measure.object_specs import OBJECT_SPECS

        text = (gen.GENERATED_DIR / "object_contracts.md").read_text(encoding="utf-8")
        for ghost in ("FitEdge", "FitRectangle", "FindCircle", "FindLine",
                      "PointToLineDistance"):
            assert f"`{ghost}`" not in text, f"{ghost} is not a real type"
        assert set(OBJECT_SPECS) == {
            "EdgePair", "EdgePoint", "FitCircle", "FitLine", "TemplateMatchPoint",
            "PointCircleDistance", "PointLineDistance", "TwoLinesAngle",
            "TwoPointsDistance", "TwoPointsLine",
        }

    def test_defaults_are_rendered_for_every_optional_param(self) -> None:
        from measure.object_specs import OBJECT_SPECS

        text = (gen.GENERATED_DIR / "object_contracts.md").read_text(encoding="utf-8")
        for spec in OBJECT_SPECS.values():
            for key in spec.defaults:
                assert f"`{key}`(" in text, f"{spec.object_type}.{key} has no default shown"


class TestV2SurfaceMatchesSnapshot:
    """The v2 table must mirror the snapshot the v2 owner publishes.

    v2 is a FastAPI service whose contract lives in ``docs/v2-openapi.json``;
    these checks guard against the generator silently rendering an empty or
    truncated surface when that file moves, shrinks or disappears.
    """

    def test_snapshot_exists_and_is_substantial(self) -> None:
        assert gen.V2_OPENAPI_SNAPSHOT.exists(), (
            "docs/v2-openapi.json missing — v2_surface.md cannot be trusted"
        )
        doc = gen._load_v2_document()
        assert len(doc.get("paths", {})) >= 50, (
            f"v2 snapshot has only {len(doc.get('paths', {}))} paths; expected the "
            f"full surface. Did the snapshot get truncated or replaced?"
        )

    def test_every_snapshot_path_appears_in_the_table(self) -> None:
        doc = gen._load_v2_document()
        text = (gen.GENERATED_DIR / "v2_surface.md").read_text(encoding="utf-8")
        for path in doc["paths"]:
            assert f"`{path}`" in text, f"{path} missing from generated surface"

    def test_openapi_paths_and_schema_counts_agree(self) -> None:
        doc = gen._load_v2_document()
        text = (gen.GENERATED_DIR / "v2_surface.md").read_text(encoding="utf-8")
        assert f"Paths: **{len(doc['paths'])}**" in text
        n_schemas = len(doc.get("components", {}).get("schemas", {}))
        assert f"Schemas: **{n_schemas}**" in text

    def test_missing_snapshot_fails_loudly(self, tmp_path) -> None:
        """A vanished snapshot must raise, not emit an empty route list."""
        original = gen.V2_OPENAPI_SNAPSHOT
        try:
            gen.V2_OPENAPI_SNAPSHOT = tmp_path / "absent.json"
            with pytest.raises(FileNotFoundError):
                gen._load_v2_document()
        finally:
            gen.V2_OPENAPI_SNAPSHOT = original
