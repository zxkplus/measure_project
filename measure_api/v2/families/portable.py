"""Upstream-owned recipe packaging and one-way compatibility calculations."""

from measure_api.v2.portable_recipes import (
    PortableRecipeResult,
    RecipeExportRequest,
    RecipeImportRequest,
    export_recipe,
    import_recipe,
    prepare_import,
    prepare_export,
)
from measure_api.v2.replay import ReplayRequest, ReplayResult, prepare_replay, replay
from measure_api.v2.legacy_conversion import (
    LegacyConversionRequest,
    LegacyConversionResult,
    convert_legacy,
    prepare_conversion,
)
from measure_api.v2.registry import Operation

OPERATIONS = [
    Operation(
        "/recipe-exports",
        RecipeExportRequest,
        PortableRecipeResult,
        export_recipe,
        "Package a verified upstream recipe and its assets",
        "recipes",
        prepare=prepare_export,
    ),
    Operation(
        "/recipe-imports",
        RecipeImportRequest,
        PortableRecipeResult,
        import_recipe,
        "Validate and relocate a supplied portable recipe",
        "recipes",
        prepare=prepare_import,
    ),
    Operation(
        "/replays",
        ReplayRequest,
        ReplayResult,
        replay,
        "Replay frozen supplied configuration with unchanged verified inputs",
        "results",
        prepare=prepare_replay,
    ),
    Operation(
        "/legacy-project-conversions",
        LegacyConversionRequest,
        LegacyConversionResult,
        convert_legacy,
        "Convert supported legacy data without GUI or pickle",
        "recipes",
        prepare=prepare_conversion,
    ),
]
