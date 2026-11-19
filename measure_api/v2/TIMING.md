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

## Counters

`counters` accumulates numeric observations that would be too numerous to keep as
spans (126 pose correlations in one operation, five re-reads of the same 20 MPx
image). Every counter is operation-local, rounded to three decimals, and bounded to
64 names; the same `truncated` flag discloses loss.

| counter | meaning |
|---|---|
| `ncc_evaluations`, `ncc_positions`, `ncc_template_pixels` | normalised cross-correlation work: evaluations, correlation positions, support pixels per evaluation |
| `ncc_masked`, `ncc_unmasked` | evaluations on the masked path (about twice the cost of the plain path) |
| `coarse_poses`, `fine_poses`, `fine_neighborhoods`, `fine_neighborhoods_refined` | pose-grid size and how many coarse neighbourhoods were rescanned |
| `pyramid_blur_ms`, `pyramid_warp_ms`, `pyramid_template_cache_hit` | decimation cost and reuse of an identical decimated support |
| `variant_render_ms`, `variant_render_count` | support rotation/scaling per pose |
| `verify_bytes`, `verify_hash_ms`, `digest_reuse` | input verification and how often an unchanged file skipped re-hashing |
| `decode_count`, `decode_ms`, `cache_hit`, `cache_miss`, `cache_write`, `cache_read_ms`, `cache_write_ms`, `cache_corrupt` | decoded-image cache behaviour |
| `memory_hit`, `image_read`, `image_bytes` | operation-local decoded-array reuse |
| `template_cache_hit`, `template_cache_miss` | validated-template reuse across operations in one worker process |
| `pose_batches_parallel` | pose-grid batches evaluated on a thread pool (`search_plan.threads` reports the configured width) |

`pyramid_prepare` is now a retained span, so the previously unaccounted
blur-and-decimate step between `preprocessing` and `coarse_search` is attributed.

Counters measure **work, not wall time**: with `MEASURE_V2_SEARCH_THREADS > 1` the
pose grid runs on a thread pool and `ncc_ms` sums the per-evaluation time of every
thread, so it can exceed the enclosing span. Compare it with
`spans.fine_search.duration_ms` to read the achieved parallelism.

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
