 #!/usr/bin/env bash
 # Measure API server startup script.
 set -euo pipefail
 
 cd "$(dirname "$0")/.."  # project root
 
 # Configuration (edit config.local.yaml or override via env)
 PORT="${MEASURE_API_PORT:-5000}"
 HOST="${MEASURE_API_HOST:-0.0.0.0}"
 CONFIG="${MEASURE_API_CONFIG:-}"
 
 echo "==> Measure API Server"
 echo "    Host: $HOST"
 echo "    Port: $PORT"
 echo ""
 
 exec python -m measure_api --host "$HOST" --port "$PORT" ${CONFIG:+--config "$CONFIG"}
