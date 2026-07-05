#!/usr/bin/env bash
set -e

echo "--- removing leftover Fogbed containers ---"
ORPHANS="$(docker ps -aq --filter 'name=^mn\.' --filter 'name=^fl_collector$' 2>/dev/null)"
if [ -n "$ORPHANS" ]; then
    docker rm -f $ORPHANS
else
    echo "(none)"
fi

echo "--- Mininet cleanup (controllers, veth, qdiscs) ---"
PYTHON_BIN="${PYTHON_BIN:-python}"
sudo "$PYTHON_BIN" -c "from mininet.clean import cleanup; cleanup()"

echo "--- done ---"
