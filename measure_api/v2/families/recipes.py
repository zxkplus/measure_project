"""Public supplied recipe validation, preview and execution operations."""

from measure_api.v2.recipe_execution import (
    execute_run,
    prepare_recipe,
    validate_snapshot,
    execute_batch,
    prepare_batch,
)
from measure_api.v2.recipe_models import (
    PreviewRequest,
    RecipeRequest,
    RecipeValidationResult,
    RunRequest,
    RunResult,
    RunBatchRequest,
    RunBatchResult,
)
from measure_api.v2.registry import CalculationContext, Operation


def validate(request: RecipeRequest, context: CalculationContext) -> RecipeValidationResult:
    """Validate an immutable explicit snapshot without saving an upstream asset.

    Args:
        request: Typed inline or file-backed upstream recipe snapshot for complete
            validation.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Configuration issues, deterministic dependency order and canonical recipe identity.
    """
    return validate_snapshot(request)


OPERATIONS = [
    Operation(
        "/recipe-validations",
        RecipeRequest,
        RecipeValidationResult,
        validate,
        "Validate a supplied typed recipe DAG",
        "recipes",
        prepare=prepare_recipe,
    ),
    Operation(
        "/recipe-previews",
        PreviewRequest,
        RunResult,
        execute_run,
        "Preview selected nodes and dependency closure",
        "recipes",
        prepare=prepare_recipe,
    ),
    Operation(
        "/run-batches",
        RunBatchRequest,
        RunBatchResult,
        execute_batch,
        "Execute complete independent recipe images",
        "recipes",
        prepare=prepare_batch,
    ),
    Operation(
        "/runs",
        RunRequest,
        RunResult,
        execute_run,
        "Execute a supplied recipe on one image",
        "recipes",
        prepare=prepare_recipe,
    ),
]
