#!/bin/bash
set -e

# Apply per-link tc rules if TC_LINK_* vars are present.
# Requires NET_ADMIN capability.
#
# Env var scheme (one set per outgoing link, N starting at 0):
#   TC_LINK_<N>_HOST      hostname of downstream node (required)
#   TC_LINK_<N>_MBPS      egress rate limit in Mbit/s (optional)
#   TC_LINK_<N>_DELAY_MS  added latency in ms (optional)
#   TC_LINK_<N>_JITTER_MS latency jitter in ms, requires DELAY_MS (optional)
#   TC_LINK_<N>_LOSS_PCT  random packet loss percentage (optional)

has_tc=false
for i in $(seq 0 9); do
    eval "host=\${TC_LINK_${i}_HOST:-}"
    if [ -n "$host" ]; then
        has_tc=true
        break
    fi
done

if [ "$has_tc" = true ]; then
    echo "[entrypoint] Configuring per-link tc rules on eth0"
    tc qdisc add dev eth0 root handle 1: htb default 99
    tc class add dev eth0 parent 1: classid 1:99 htb rate 1000mbit

    for i in $(seq 0 9); do
        eval "HOST=\${TC_LINK_${i}_HOST:-}"
        [ -z "$HOST" ] && break

        eval "MBPS=\${TC_LINK_${i}_MBPS:-}"
        eval "DELAY_MS=\${TC_LINK_${i}_DELAY_MS:-}"
        eval "JITTER_MS=\${TC_LINK_${i}_JITTER_MS:-}"
        eval "LOSS_PCT=\${TC_LINK_${i}_LOSS_PCT:-}"

        # Resolve hostname to IP with retries
        IP=""
        for attempt in 1 2 3 4 5; do
            IP=$(getent hosts "$HOST" | awk '{print $1; exit}')
            [ -n "$IP" ] && break
            echo "[entrypoint] Waiting for DNS resolution of $HOST (attempt $attempt/5)..."
            sleep 2
        done

        if [ -z "$IP" ]; then
            echo "[entrypoint] WARNING: could not resolve $HOST after 5 attempts, skipping tc rule"
            continue
        fi

        CLASSID="1:$((i + 1))"
        HANDLE="$((i + 10)):"

        # HTB class — rate limit if MBPS set, else pass-through at line rate
        if [ -n "$MBPS" ]; then
            tc class add dev eth0 parent 1: classid "$CLASSID" htb rate "${MBPS}mbit"
        else
            tc class add dev eth0 parent 1: classid "$CLASSID" htb rate 1000mbit
        fi

        # netem leaf qdisc for delay / loss if either is set
        if [ -n "$DELAY_MS" ] || [ -n "$LOSS_PCT" ]; then
            NETEM_ARGS=""
            if [ -n "$DELAY_MS" ]; then
                NETEM_ARGS="delay ${DELAY_MS}ms"
                [ -n "$JITTER_MS" ] && NETEM_ARGS="$NETEM_ARGS ${JITTER_MS}ms"
            fi
            [ -n "$LOSS_PCT" ] && NETEM_ARGS="$NETEM_ARGS loss ${LOSS_PCT}%"
            tc qdisc add dev eth0 parent "$CLASSID" handle "$HANDLE" netem $NETEM_ARGS
        fi

        # u32 filter: steer traffic to this destination into the class
        tc filter add dev eth0 parent 1: protocol ip u32 \
            match ip dst "${IP}/32" flowid "$CLASSID"

        echo "[entrypoint] tc: $HOST ($IP)${MBPS:+ rate=${MBPS}mbit}${DELAY_MS:+ delay=${DELAY_MS}ms}${JITTER_MS:+ jitter=${JITTER_MS}ms}${LOSS_PCT:+ loss=${LOSS_PCT}%}"
    done
fi

exec "$@"
