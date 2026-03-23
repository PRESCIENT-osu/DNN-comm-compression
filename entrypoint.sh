#!/bin/bash
set -e

# Apply egress bandwidth limit via tc netem if TC_BANDWIDTH_MBPS is set.
# Requires NET_ADMIN capability (set in docker-compose / pod spec).
if [ -n "$TC_BANDWIDTH_MBPS" ] && [ "$TC_BANDWIDTH_MBPS" != "0" ]; then
    echo "[entrypoint] Applying egress bandwidth limit: ${TC_BANDWIDTH_MBPS} Mbit/s on eth0"
    tc qdisc add dev eth0 root netem rate "${TC_BANDWIDTH_MBPS}mbit" 2>/dev/null || \
        echo "[entrypoint] Warning: tc qdisc add failed (may already be set)"
fi

exec "$@"
