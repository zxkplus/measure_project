# Operation timing diagnostics

All V2 calculations return additive timing metadata in the existing free-form
`Operation.diagnostics.timing` object. The public OpenAPI schema is unchanged.
The original `timing_version=3` metrics remain available and keep their meanings.

`schema_version` is 1. `status` is `complete` or `partial`; `total_ms` measures the
runtime admission-to-terminal interval. `spans` contain `id`, optional `parent_id`,
`name`, `kind`, `start_ms`, optional `duration_ms`, and `state`. Recipe spans also
carry `node_id`, `target_id` and `space` where applicable. Every offset is relative
to the operation start in the shared OS monotonic clock domain; absolute clock
readings are not part of this new object.

Durations include child stages. Do not add execution to its nodes, or add nodes to
their sampling/fitting stages. Pixel and physical executions use distinct span IDs.
`kind` distinguishes phase, target, node and step. Runtime queue, worker dispatch,
preparation, identity binding, execution and terminal persistence are separate
observations. File verification, image decoding, preprocessing, template coarse/fine
search, edge sampling, geometry fitting, result validation/encoding and artifact
commit appear when those paths execute.

A running span has no final `duration_ms`. Cancellation, timeout or worker death
retain the last known running phase without inventing its end. The complete terminal
persistence duration is available only after that write returns. Durable recovery
may therefore expose a partial tail even for a successful operation; this never
changes the durable business result.

At most 500 spans and 64 aggregate names are retained. Slow/error leaves are preferred
after the span budget fills, while retained parents stay available. `truncated=true`
and `status=partial` disclose loss. Aggregates report count, inclusive `total_ms` and
`max_ms` by name and kind; aggregate rows may overlap and must not be summed together.
Progress sends bounded individual phase/node/target updates, never a growing full
snapshot per tool. Instrumentation uses a context-local recorder; ordinary calls
outside a V2 operation keep the same outputs and exceptions.

Validation: `python -m pytest tests/v2/test_timing.py tests/v2/test_operation_lifecycle.py tests/v2/test_recipes.py`.
