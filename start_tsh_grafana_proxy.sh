#!/bin/bash

set -euo pipefail

# Kill any existing tsh proxy processes -f here isn't force its match the full string
pkill -f "tsh proxy app grafana" 2>/dev/null || true

if nc -z localhost 8080 2>/dev/null; then
    echo "Error: Port 8080 is still in use by another process" >&2
    exit 1
fi

# Start TSH proxy and immediately return PID
tsh proxy app grafana --port=8080 >/dev/null 2>&1 &
echo $!