# V2 Production Measurement Engine

The production process exposes only the versioned `/api/v2` contract. The
legacy `/api/session` server remains a development interface and is not used by
the production entry point.

## Start

```bash
MEASURE_V2_ASSET_ROOT=/var/lib/measure-v2 \
MEASURE_V2_SHM_NAME=measure_v2_img_shm \
python3 -m measure_api.v2
```

The default internal port is `9004`. `GET /api/v2/health` reports only whether
the HTTP process can accept requests. The static object, image, timeout, error,
and operation contract is available at `GET /api/v2/openapi.json`.

## Deployment invariants

- Deploy one platform writer and one v2 engine instance on the same node.
- Bind the same host `/dev/shm` into both containers.
- Reserve the fixed `measure_v2_img_shm` pool for v2. It has 30 slots of
  20 MiB each (629,145,600 bytes total) and must not be shared with v1.
- Send JPEG only. A slot reference is temporary and valid only until v2 has
  decoded and copied a private contiguous grayscale snapshot.
- Mount `/var/lib/measure-v2` on persistent node storage. The directory is the
  engine-owned authority for current model assets and exported packages.
- Model publication directly overwrites the current asset. Cancellation,
  timeout, or write failure after overwrite starts can leave it unavailable.

## Accepted first-release limits

Operations run serially through one replaceable CPU subprocess. POST requests
are neither retried nor idempotent, concurrent state changes are unsupported,
and operations do not survive process restart. Shared memory has no lease,
generation, checksum, completion marker, or timeout quarantine. Model assets
have no automatic rollback, node-external backup, orphan cleanup, or cross-node
replication. The engine never falls back to another model or engine.

The container smoke test should verify process startup, health, a
629,145,600-byte v2 shared-memory object, asset write/read persistence, and a
deterministic JPEG measurement. Protocol regression tests do not constitute
approval of production accuracy, repeatability, or throughput.
