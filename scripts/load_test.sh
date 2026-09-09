#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Same 60-second/16-client load as the supplied hey command, with persisted metrics.
if [[ -x .venv/bin/python ]]; then
  .venv/bin/python scripts/http_load_test.py --duration 60 --concurrency 16
else
  python scripts/http_load_test.py --duration 60 --concurrency 16
fi
